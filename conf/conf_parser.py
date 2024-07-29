import json
import yaml
import os.path

from algorithms.algorithms_utils import AlgorithmsEnum
from data.config_classes import ExperimentConfig, DatasetSplitType, DatasetsEnum
from data.data_utils import merge_dicts
from data_paths import get_dataset_path, get_results_path
from utilities.utils import generate_id


def get_config(config: str | dict, alg: AlgorithmsEnum, dataset: DatasetsEnum,
               split_type: DatasetSplitType, dataset_path: str = None,
               run_id: str = None) -> ExperimentConfig:
    # load or simply use provided config
    config_dict = load_config_dict(config) if isinstance(config, str) else config

    # make sure to inform user about code changes
    warn_parameter_ignored_if_present(config_dict, 'algorithm')
    warn_parameter_ignored_if_present(config_dict, 'data_path')
    warn_parameter_ignored_if_present(config_dict, 'dataset_path')
    warn_parameter_ignored_if_present(config_dict, 'wandb.wandb_path')

    # run has specific id
    run_id = run_id or generate_id()

    # retrieve correct dataset path and store it in the config
    dataset_path = dataset_path or get_dataset_path(dataset, split_type)
    is_cold_start_dataset = split_type in [DatasetSplitType.ColdStartUser,
                                           DatasetSplitType.ColdStartItem,
                                           DatasetSplitType.ColdStartBoth]
    update_nested_dict(config_dict, 'dataset.dataset_path', dataset_path)
    update_nested_dict(config_dict, 'dataset.is_cold_start_dataset', is_cold_start_dataset)

    # retrieve different paths (directories) for results
    results_path, wandb_path = get_and_create_results_paths(alg, dataset, split_type, config_dict, run_id)

    # store wandb path in config
    update_nested_dict(config_dict, 'wandb.wandb_path', wandb_path)

    # finish by returning the full and processed configuration
    return ExperimentConfig.from_dict_ext(
        config_dict,
        dict_has_precedence=False,
        run_id=run_id,
        algorithm_type=alg,
        dataset_type=dataset,
        split_type=split_type,
        results_path=results_path
    )


def get_and_create_results_paths(alg: AlgorithmsEnum, dataset: DatasetsEnum, split_type: DatasetSplitType,
                                 config_dict: dict, run_id: str):
    # retrieve base path for the results
    base_results_path = get_results_path()

    # base folder should describe what kind of category the experiment falls into
    run_sub_folder_tree = [f'{alg}-{dataset}-{split_type}']

    # decide on the folder structure based on whether and if, what kind of hyperparameter search we are doing
    if not nested_dict_get(config_dict, 'run_settings.in_tune', False):
        if sweep_id := nested_dict_get(config_dict, 'wandb.sweep_id'):
            run_sub_folder_tree += ['sweeps', sweep_id]
        else:
            run_sub_folder_tree += ['single_runs']
    run_sub_folder_tree += [run_id]

    # store model and wandb data separately for easier deletion
    wandb_path = os.path.join(base_results_path, 'wandb', *run_sub_folder_tree)
    results_path = os.path.join(base_results_path, 'results', *run_sub_folder_tree)

    # ensure that directories already exist
    os.makedirs(wandb_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    return results_path, wandb_path


def warn_parameter_ignored_if_present(config_dict: dict, param_name: str):
    if nested_dict_get(config_dict, param_name) is not None:
        print(f'Specifying "{param_name}" in config file is deprecated and will thus be ignored.')


def raise_on_config_mismatch(first, second, name):
    if first != second:
        raise ValueError(f'Specified {name} does not match the {name} in the config file.')


def update_nested_dict(d: dict, key: str, value: any):
    keys = key.split('.')
    current_dict = d
    for k in keys[:-1]:
        current_dict = current_dict.setdefault(k, {})
    current_dict[keys[-1]] = value


def nested_dict_get(d: dict, key: str, default_value: any = None):
    keys = key.split('.')
    current_dict = d
    for k in keys[:-1]:
        next_dict = current_dict.get(k)
        if not isinstance(next_dict, dict):
            return default_value
        current_dict = next_dict
    return current_dict.get(keys[-1], default_value)


def load_config_dict(config_path: str, ignore_base_configs=False):
    config = parse_conf_file(config_path)
    if not ignore_base_configs:
        config = extend_by_base_configs(config, config_path)
    return config


def extend_by_base_configs(config, config_path=None):
    # to have more concise config files, we'll allow extending basic configurations
    # multiple files can extend one another by always specifying what their basis is
    all_base_config = {}
    if base_config_paths := config.get('base_configs'):
        if isinstance(base_config_paths, str):
            base_config_paths = [base_config_paths]

        for base_config_path in base_config_paths:
            # allow to specify absolut paths, which might be necessary at times
            if not os.path.isabs(base_config_path) and config_path is not None:
                # assumes that the base config file is located in the same directory as the actual config file
                # as it's path was specified
                base_config_path = os.path.join(os.path.dirname(config_path), base_config_path)

            base_config = load_config_dict(base_config_path)
            # base configs have precedence the further down they are listed
            all_base_config = merge_dicts(all_base_config, base_config)

    # config has always a higher precedence than its base configs
    config = merge_dicts(all_base_config, config)
    return config


def parse_conf_file(config_path: str) -> dict:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Configuration file "{config_path}" not found!')

    _, ext = os.path.splitext(config_path)
    with open(config_path, 'r') as fh:
        match ext:
            case '.yml' | '.yaml':
                return yaml.safe_load(fh)
            case '.json':
                return json.load(fh)
            case _:
                raise ValueError(f'Config file of type "{ext}" are not supported. Supply either a .yml or .json file.')


def yaml_save(file_path: str, data: any):
    with open(file_path, 'w') as fh:
        yaml.dump(data, fh)


def save_config(conf_path: str, conf: dict):
    conf_path = os.path.join(conf_path, 'conf.yml')
    return yaml_save(conf_path, conf)
