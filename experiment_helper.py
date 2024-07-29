import json
import os
import wandb
import typing

from algorithms.algorithms_utils import AlgorithmsEnum, get_algorithm_class
from algorithms.base_classes import SGDBasedRecommenderAlgorithm, SparseMatrixBasedRecommenderAlgorithm
from conf.conf_parser import parse_conf_file, save_config, get_config, raise_on_config_mismatch, yaml_save
from data.config_classes import DatasetSplitType, ExperimentConfig, DatasetsEnum
from data.data_utils import get_dataset_and_loader
from eval.eval import evaluate_recommender_algorithm, FullEvaluator, gather_recommender_algorithm_results
from train.rec_losses import RecommenderSystemLoss
from train.trainer import Trainer
from utilities.utils import reproducible
from wandb_conf import PROJECT_NAME, ENTITY_NAME


def run_train_val_experiment(alg: AlgorithmsEnum, dataset: DatasetsEnum, split_type: DatasetSplitType,
                             conf: typing.Union[str, dict, ExperimentConfig], dataset_path: str = None):

    if not isinstance(conf, ExperimentConfig):
        conf = get_config(conf, alg, dataset, split_type, dataset_path)

    if conf.wandb.use_wandb:
        tags = [conf.algorithm_name, conf.dataset_name, conf.split_name]
        group_tags = tags + ['train/val']
        wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME, config=conf.to_dict(),
                   tags=tags, group=' - '.join(group_tags), name=conf.run_id,
                   job_type='train/val', dir=conf.wandb.wandb_path)

    # do the actual training
    metrics_values = run_train_val(conf)
    print('best validation results', json.dumps(metrics_values, indent='\t'))

    # finish experiment
    if conf.wandb.use_wandb:
        wandb.finish()

    return metrics_values, conf


def run_train_val(conf: ExperimentConfig):
    print(f'Starting a train & validation experiment with '
          f'"{conf.algorithm_name}" algorithm on {conf.split_type} "{conf.dataset_type}" dataset')

    # store config for run
    save_config(conf.results_path, conf.to_dict())

    # ensure reproducibility of experiments
    reproducible(conf.run_settings.seed)

    train_set, train_loader = get_dataset_and_loader(conf, 'train')
    val_set, val_loader = get_dataset_and_loader(conf, 'val')

    # optionally get the validation loader for the train set
    _, train_val_loader = (None, None) if conf.train_eval is None else get_dataset_and_loader(conf, 'train',
                                                                                              retrieve_eval_loader=True)

    alg_cls = get_algorithm_class(conf.algorithm_type)
    alg_instance = alg_cls.build_from_conf(conf.model, train_set)

    print(f'Training algorithm {conf.algorithm_name}:\n{alg_instance}\n')

    if issubclass(alg_cls, SGDBasedRecommenderAlgorithm):
        rec_loss = RecommenderSystemLoss.build_from_conf(conf, train_loader.dataset)
        trainer = Trainer(alg_instance, train_loader, val_loader, rec_loss, conf, train_val_loader=train_val_loader)
        # Note that the trainer instance returns the metrics of the epoch where the best
        # metric was achieved. Thus, when logging, the last step isn't an actual step,
        # but a replicate of the values when the best metric was achieved.
        metrics_values = trainer.fit()

    elif issubclass(alg_cls, SparseMatrixBasedRecommenderAlgorithm):
        alg_instance.fit(matrix=train_set.user_sampling_matrix)
        evaluator = FullEvaluator(config=conf.eval, dataset=val_set)
        metrics_values = evaluate_recommender_algorithm(alg_instance, val_loader, evaluator,
                                                        verbose=conf.run_settings.batch_verbose)
        alg_instance.save_model_to_path(conf.results_path)

    elif conf.algorithm_type in [AlgorithmsEnum.rand, AlgorithmsEnum.pop]:
        evaluator = FullEvaluator(config=conf.eval, dataset=val_set)
        metrics_values = evaluate_recommender_algorithm(alg_instance, val_loader, evaluator,
                                                        verbose=conf.run_settings.batch_verbose)
    else:
        raise ValueError(f'Training for "{alg_cls}" has been not implemented')

    if conf.wandb.use_wandb:
        wandb.log(metrics_values)

    yaml_save(os.path.join(conf.results_path, 'metrics_val.yml'), metrics_values)
    return metrics_values


def run_test_experiment(alg: AlgorithmsEnum, dataset: DatasetsEnum, split_type: DatasetSplitType,
                        conf: typing.Union[str, dict, ExperimentConfig]):
    print(f'Starting a test experiment with "{alg}" algorithm on {split_type} "{dataset}" dataset')

    if not isinstance(conf, ExperimentConfig):
        config_dict = parse_conf_file(conf) if isinstance(conf, str) else conf
        conf = ExperimentConfig.from_dict(config_dict)

    raise_on_config_mismatch(alg, conf.algorithm_type, 'algorithm')
    raise_on_config_mismatch(dataset, conf.dataset_type, 'dataset')
    raise_on_config_mismatch(split_type, conf.split_type, 'split type')

    if conf.wandb.use_wandb:
        tags = [conf.algorithm_name, conf.dataset_name, conf.split_name]
        group_tags = tags + ['test']

        wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME, config=conf.to_dict(), tags=tags,
                   group=' - '.join(group_tags), name=conf.run_id, job_type='test', reinit=True)

    # do the actual training
    metrics_values = run_test(conf)
    print('test results', json.dumps(metrics_values, indent='\t'))

    # finish experiment
    if conf.wandb.use_wandb:
        wandb.finish()

    return metrics_values


def run_test(conf: ExperimentConfig, store_results: bool = True, return_raw=False):
    train_set, _ = get_dataset_and_loader(conf, 'train')
    test_set, test_loader = get_dataset_and_loader(conf, 'test')

    alg_cls = get_algorithm_class(conf.algorithm_type)
    if conf.algorithm_type in [AlgorithmsEnum.pop, AlgorithmsEnum.dmf, AlgorithmsEnum.ecf]:
        # some algorithms, e.g. pop recommender, require access to training data
        alg_instance = alg_cls.build_from_conf(conf.model, train_set)
    else:
        alg_instance = alg_cls.build_from_conf(conf.model, test_set)

    alg_instance.load_model_from_path(conf.results_path)
    evaluator = FullEvaluator(config=conf.eval, evaluator_name='test', dataset=test_set)
    results = evaluate_recommender_algorithm(alg_instance, test_loader, evaluator, return_raw=return_raw,
                                             verbose=conf.run_settings.batch_verbose)
    metric_results = results[0] if isinstance(results, tuple) else results
    if conf.wandb.use_wandb:
        # need to define a custom x-axis for test metrics, as logging to step 0 does not work
        # for already running wandb runs
        # see https://docs.wandb.ai/guides/track/log/customize-logging-axes
        wandb.define_metric("test/step")
        wandb.define_metric("test/*", step_metric="test/step")
        wandb.log(metric_results)

    if store_results:
        yaml_save(os.path.join(conf.results_path, 'metrics_test.yml'), metric_results)
    return results


def run_gather_experiment(alg: AlgorithmsEnum, dataset: DatasetsEnum, split_type: DatasetSplitType,
                          conf: typing.Union[str, dict, ExperimentConfig]):
    print(f'Starting a gather experiment with "{alg}" algorithm on {split_type} "{dataset}" dataset')

    if not isinstance(conf, ExperimentConfig):
        config_dict = parse_conf_file(conf) if isinstance(conf, str) else conf
        conf = ExperimentConfig.from_dict(config_dict)

    raise_on_config_mismatch(alg, conf.algorithm_type, 'algorithm')
    raise_on_config_mismatch(dataset, conf.dataset_type, 'dataset')
    raise_on_config_mismatch(split_type, conf.split_type, 'split type')

    # do the actual gathering
    return run_gather(conf)


def run_gather(conf: ExperimentConfig, results_file: str = None, split: str = 'test'):
    train_set, _ = get_dataset_and_loader(conf, 'train')
    test_set, test_loader = get_dataset_and_loader(conf, split)

    alg_cls = get_algorithm_class(conf.algorithm_type)
    if conf.algorithm_type in [AlgorithmsEnum.pop, AlgorithmsEnum.dmf, AlgorithmsEnum.ecf]:
        # some algorithms, e.g. pop recommender, require access to training data
        alg_instance = alg_cls.build_from_conf(conf.model, train_set)
    else:
        alg_instance = alg_cls.build_from_conf(conf.model, test_set)

    alg_instance.load_model_from_path(conf.results_path)
    if isinstance(alg_instance, SGDBasedRecommenderAlgorithm):
        alg_instance.eval()

    evaluator = FullEvaluator(config=conf.eval, evaluator_name=split, dataset=test_set)
    results = gather_recommender_algorithm_results(alg_instance, test_loader, evaluator,
                                                   results_path=results_file)
    return results


def run_train_val_test(conf: ExperimentConfig):
    _ = run_train_val(conf)
    return run_test(conf)


def run_train_val_test_experiment(alg: AlgorithmsEnum, dataset: DatasetsEnum, split_type: DatasetSplitType,
                                  conf: typing.Union[str, dict, ExperimentConfig], dataset_path: str = None):
    print(f'Starting a train, validation & test experiment with "{alg}" algorithm on {split_type} "{dataset}" dataset')
    _, conf = run_train_val_experiment(alg, dataset, split_type, conf, dataset_path)
    return run_test_experiment(alg, dataset, split_type, conf)
