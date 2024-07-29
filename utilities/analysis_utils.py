import os
from glob import glob
from pathlib import Path

import wandb
import dill as pkl
import pandas as pd
from collections import defaultdict
from tqdm.autonotebook import tqdm
from timeout_decorator import timeout, TimeoutError
from data_paths import get_results_path, _base_server_results_path_map, get_dataset_path
from data.config_classes import ExperimentConfig

from data.data_utils import get_dataset_and_loader
from algorithms.algorithms_utils import get_algorithm_class
import yaml


def flatten_dictionary(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary.

    Args:
    - d: The dictionary to flatten.
    - parent_key (str): The current parent key in the recursion.
    - sep (str): The separator to use between keys.

    Returns:
    - dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dictionary(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def retrieve_runs(project_name):
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(project_name, per_page=300, filters={'state': 'finished'})

    summary_list, config_list, description_list = [], [], []
    for run in tqdm(runs, desc='fetching runs from W&B'):
        # gather useful general information about the run
        description_list.append({
            'run_id': run.id,
            'name': run.name,
            'state': run.state,
            'tags': run.tags
        })

        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items()
             if not k.startswith('_')})

    fc = [flatten_dictionary(c) for c in config_list]
    fs = [flatten_dictionary(s) for s in summary_list]
    fd = [flatten_dictionary(d) for d in description_list]
    r = [{**d, **c, **s} for d, s, c in zip(fd, fs, fc)]

    df = pd.DataFrame.from_dict(r)
    df = df.rename(columns={'wandb.sweep_id': 'sweep_id'})

    # change column order
    first_columns = ['sweep_id', 'run_id']
    column_order = first_columns + [c for c in df.columns if c not in first_columns]
    df = df[column_order]

    return df


def get_run_host(project_name, run_id):
    api = wandb.Api()
    return api.run(f'{project_name}/{run_id}').metadata['host']


class DataStore:
    def __init__(self, storage_file=None):
        self.storage_file = storage_file

        # have two data collections
        # -- one for simply data that's only looked up by a key
        self._data = defaultdict(lambda: None)
        # -- and one that is a nested dictionary, e.g., where for a key, some sub keys may exist
        self._dict_data = defaultdict(lambda: defaultdict())

        if os.path.exists(self.storage_file):
            self._load_store()

    def reset(self):
        self._data = defaultdict(lambda: None)
        self._dict_data = defaultdict(lambda: defaultdict())

        if os.path.exists(self.storage_file):
            os.remove(self.storage_file)

    def _load_store(self):
        with open(self.storage_file, 'rb') as fh:
            obj = pkl.load(fh)

        self._data = defaultdict(lambda: None)
        self._dict_data = defaultdict(lambda: defaultdict())

        data, dict_data = obj
        self._data.update(data)
        for k, v in dict_data.items():
            self._dict_data[k].update(v)

    def _save_store(self):
        with open(self.storage_file, 'wb') as fh:
            # transform to dict to only use low level types with pkl (avoid possible deserialization problems)
            data = dict(self._data)
            dict_data = {k: dict(v) for k, v in self._dict_data.items()}
            return pkl.dump((data, dict_data), fh)

    def exists(self, store_key):
        return store_key in self._data

    def exists_nested(self, store_key, nested_key):
        return store_key in self._dict_data and nested_key in self._dict_data[store_key]

    def update(self, store_key, value, flush=True):
        self._data[store_key] = value

        if flush:
            self._save_store()

    def flush(self):
        self._save_store()

    def update_nested(self, store_key, flush=True, **kwargs):
        for k, v in kwargs.items():
            self._dict_data[store_key][k] = v

        if flush:
            self._save_store()

    def get(self, store_keys: str | list[str], default=None):
        single = isinstance(store_keys, str)
        if single:
            store_keys = (store_keys,)

        r = tuple(self._data.get(key, default) for key in store_keys)
        return r[0] if single else r

    def get_nested(self, store_key: str, nested_keys: str | list[str], default=None):
        # do prep work to handle all arguments combinations the same way
        single_nested = isinstance(nested_keys, str)

        if single_nested:
            nested_keys = (nested_keys,)

        if store_key in self._dict_data:
            r = tuple(self._dict_data[store_key].get(nk, default) for nk in nested_keys)
        else:
            r = (default,) * len(nested_keys)

        # postprocess data to match argument formats
        if single_nested:
            return r[0]
        return r


@timeout(1, use_signals=False)
def isdir_with_timeout(s):
    return os.path.isdir(s)


def get_all_storage_paths(verbose=True):
    if verbose:
        print('collecting all storage paths')

    all_possible_paths = []
    raise NotImplementedError('add your logic here')

    # storage_paths = []
    # # only keep paths that do exist (and are accessible)
    # for p in all_possible_paths:
    #     try:
    #         if isdir_with_timeout(p):
    #             storage_paths.append(p)
    #     except TimeoutError:
    #         print(f'could not check whether "{p}" exists (ran into timeout)')
    # return storage_paths


@timeout(5, use_signals=False)
def search_with_timeout(*args, **kwargs):
    return glob(*args, **kwargs)


def create_run_lookup(additional_storage_paths: tuple[str] = (), verbose=True):
    run_dir_lookup = {}

    paths = get_all_storage_paths(verbose) + list(additional_storage_paths)

    # show all available paths that might contain run data
    if verbose:
        print('searching in the following paths for results')
        print('found the following paths:')
        print(paths)

    for p in tqdm(paths, 'searching through possible results locations'):
        try:
            run_result_files = search_with_timeout(os.path.join(p, 'results', '**', 'conf.yml'), recursive=True)
            partial_lookup = {Path(f).parts[-2]: f for f in run_result_files}
            run_dir_lookup.update(partial_lookup)
        except TimeoutError:
            print(f'glob timeout occurred for path "{p}"')

    if verbose:
        print(f'found {len(run_dir_lookup)} result directories')
    return run_dir_lookup


def load_experiment_config(run_file, eval_on_gender: bool = True):
    with open(run_file, 'r') as fh:
        a = yaml.safe_load(fh)

    # as we might access files from different servers, we'll adjust the results path accordingly
    file_dir, file_name = os.path.split(run_file)
    a['results_path'] = file_dir

    if eval_on_gender:
        # ensure that gender is one of the features in the dataset
        gender_feature = {'name': 'gender', 'type': 'categorical'}
        if ufd := a['dataset'].get('user_feature_definitions'):
            if not any(d['name'] == 'gender' for d in ufd):
                ufd.append(gender_feature)
        else:
            a['dataset']['user_feature_definitions'] = [gender_feature]

    # just load the full dataset with all the features to ensure that we have everything we might need
    conf = ExperimentConfig.from_dict(a)

    # change dataset path to current machine (in case it was not run on our servers)
    conf.dataset['dataset_path'] = get_dataset_path(conf.dataset_type, conf.split_type)
    conf.dataset['model_requires_pop_distribution'] = True
    conf.dataset['model_requires_item_interactions'] = True
    conf.dataset['model_requires_train_interactions'] = True
    conf.dataset['use_dataset_negative_sampler'] = True

    conf.wandb.use_wandb = False

    conf.eval.calculate_group_metrics = eval_on_gender
    conf.eval.user_group_features = ['gender'] or None

    # conf.eval.metrics = ['ndcg', 'precision', 'recall', 'f_score', 'hitrate', 'coverage']
    conf.eval.metrics = ['ndcg', 'precision', 'recall', 'f_score', 'hitrate', 'coverage', 'ap']

    return conf


def load_setup(conf, split='test'):
    dataset, loader = get_dataset_and_loader(conf, split, True)
    alg_cls = get_algorithm_class(conf.algorithm_type)
    alg_instance = alg_cls.build_from_conf(conf.model, dataset)
    alg_instance.load_model_from_path(conf.results_path)
    return dataset, loader, alg_instance
