import os
import glob
import json
import wandb

from data_paths import get_results_path
from experiment_helper import run_train_val_test
from data.config_classes import AlgorithmsEnum, DatasetsEnum, DatasetSplitType
from conf.conf_parser import extend_by_base_configs, get_config, update_nested_dict


def train_val_agent():
    results_path = get_results_path()

    # initialization and gathering hyperparameters
    run = wandb.init(job_type='train/val/test', allow_val_change=True, dir=results_path)

    run_id = run.id
    project = run.project
    entity = run.entity
    sweep_id = run.sweep_id

    # retrieve config for run (this already contains the hyperparameter search modifications from W&B
    conf = {k: v for k, v in wandb.config.items() if k[0] != '_'}

    print('=' * 80)
    print('W&B provided configuration is\n', json.dumps(conf, indent=4))
    print('=' * 80)

    # need to pop some parameters from the dictionary to avoid causing problems
    alg = AlgorithmsEnum(conf.pop('algorithm_type'))
    dataset = DatasetsEnum(conf.pop('dataset_type'))
    split_type = DatasetSplitType(conf.pop('split_type'))
    dataset_path = conf.pop('dataset_path', None)

    # use base configurations as well (simplifies sweep config files)
    conf = extend_by_base_configs(conf)

    update_nested_dict(conf, 'wandb.sweep_id', sweep_id)
    update_nested_dict(conf, 'wandb.use_wandb', True)

    # get full config
    conf = get_config(conf, alg, dataset, split_type, dataset_path, run_id=run_id)

    print('=' * 80)
    print('Final config is\n', json.dumps(conf.to_dict(), indent=4))
    print('=' * 80)

    # updating wandb data
    run.tags += (conf.algorithm_name, conf.dataset_name, conf.split_name)

    # make wandb aware of the whole config we are using for the run
    # it is okay if this writes warnings about unsuccessful updates to stdout
    # (does so, even if actual values don't change)
    wandb.config.update(conf.to_dict())

    print(f'W&B sweep ID is "{sweep_id}"')
    run_train_val_test(conf)

    # To reduce space consumption. Check if the run is in the top-10 best. If not, delete the model.
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    top_runs = api.runs(path=f'{entity}/{project}',
                        per_page=conf.wandb.keep_top_runs,
                        order=sweep.order,
                        filters={"$and": [{"sweep": f"{sweep_id}"}]}
                        )[:conf.wandb.keep_top_runs]
    top_runs_ids = {x.id for x in top_runs}

    if run_id not in top_runs_ids:
        print(f'Run {run_id} is not in the top-{conf.wandb.keep_top_runs}.'
              f'Model will be deleted')

        # delete local run files
        alg_model_path = os.path.join(conf.results_path, 'model.*')
        alg_model_files = glob.glob(alg_model_path)
        for alg_model_file in alg_model_files:
            os.remove(alg_model_file)

    wandb.finish()


if __name__ == '__main__':
    train_val_agent()
