import os
import platform

from data.config_classes import DatasetSplitType, DatasetsEnum

# for multiple developers, create mapping of our computer names to data locations
# you can figure out your computer name by
# >>> import platform
# >>> platform.node()
_base_local_dataset_path_map = {
    '<your-computer-name>': '<your-dataset-path>'
}

_base_local_results_path_map = {
    '<your-computer-name>': '<your-results-path>'
}

# as all developers work on the same servers, instead of mapping computer names to locations,
# we map usernames to locations. You can list your username by
# >>> import os
# >>> os.getlogin()
_base_server_dataset_path_map = {
    "<your-server-username>": "<your-server-dataset-path>"
}
_base_server_results_path_map = {
    "<your-server-username>": "<your-server-results-path>"
}

relative_data_paths = {
    DatasetsEnum.onion: 'onion',
    DatasetsEnum.onion18: 'onion18',
    DatasetsEnum.onion18g: 'onion18g',
    DatasetsEnum.ml1m: 'ml-1m',
    DatasetsEnum.ml100k: 'ml-100k',
    DatasetsEnum.kuai: 'kuai',
    DatasetsEnum.amazonvid2024: 'amazonvid2024'
}

split_type_to_dir_name = {
    DatasetSplitType.Random: 'random_split',
    DatasetSplitType.Temporal: 'temporal_split',
    DatasetSplitType.ColdStartUser: 'cold_start_user',
    DatasetSplitType.ColdStartItem: 'cold_start_item',
    DatasetSplitType.ColdStartBoth: 'cold_start_both'
}


def _is_running_on_server():
    return False


def _get_dataset_root_path(dataset: DatasetsEnum):
    # determine whether we are running on server
    if _is_running_on_server():
        username = os.getlogin()
        if username not in _base_server_dataset_path_map:
            raise KeyError(f"No dataset location found for user '{username}' on server. "
                           f"Please extend '_base_server_dataset_path_map' in 'data_paths.py'.")
        path = os.path.join(_base_server_dataset_path_map[username], relative_data_paths[dataset])
    else:
        computer_name = platform.node()
        if computer_name not in _base_local_dataset_path_map:
            raise KeyError(f"No dataset location found on computer '{computer_name}'. "
                           f"Please extend '_base_local_dataset_path_map' in 'data_paths.py'.")
        path = os.path.join(_base_local_dataset_path_map[computer_name], relative_data_paths[dataset])

    return path


def get_dataset_path(dataset: DatasetsEnum, split_type: DatasetSplitType):
    return os.path.join(get_processed_dataset_path(dataset), split_type_to_dir_name[split_type])


def get_raw_dataset_path(dataset: DatasetsEnum):
    return os.path.join(_get_dataset_root_path(dataset), 'raw_dataset')


def get_processed_dataset_path(dataset: DatasetsEnum):
    return os.path.join(_get_dataset_root_path(dataset), 'processed_dataset')


def get_results_path():
    # determine whether we are running on server
    if _is_running_on_server():
        username = os.getlogin()
        if username not in _base_server_results_path_map:
            raise KeyError(f"No results location found for user '{username}' on server. "
                           f"Please extend '_base_server_results_path_map' in 'data_paths.py'.")
        path = _base_server_results_path_map[username]
    else:
        computer_name = platform.node()
        if computer_name not in _base_local_results_path_map:
            raise KeyError(f"No results location found on computer '{computer_name}'. "
                           f"Please extend '_base_local_results_path_map' in 'data_paths.py'.")
        path = _base_local_results_path_map[computer_name]
    return path
