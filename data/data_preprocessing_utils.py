import os
import enum
import math
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.config_classes import FeatureType, FeatureBaseDefinition
from data.preprocessing_data_classes import EntityFeatures, SplitData, AllSplitsData, RawDataset, MultiDFeature
from data.filtering import print_description_listening_history, filter_based_on_indices, print_verbose, \
    sort_based_on_indices
from data.preprocessing_config_classes import DataPreprocessingConfig, SplitConfig, ColdStartType, SplitType

SPLIT_NAMES = ('train', 'val', 'test')
LOG_FILT_DATA_PATH = "log_filtering_data.txt"


def print_and_log(log_file, n_lhs, n_users, n_items, text):
    """
    Prints to screen and logs to file statistics of the data during the processing steps.
    @param log_file: Name of the file to log information to
    @param n_lhs: Number of listening events
    @param n_users: Number of users
    @param n_items: Number of items
    @param text: Text to be added
    @return:
    """
    info_string = "{:10d} entries {:7d} users {:7d} items for {}".format(n_lhs, n_users, n_items, text)
    log_file.write(info_string + '\n')
    print(info_string)


def k_core_filtering(lhs: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Performs core-filtering on the dataset.
    @param lhs: Pandas Dataframe containing the listening records. Has columns ["user", "item"]
    @param k: Threshold for performing the k-core filtering.
    @return: Filtered Dataframe
    """
    while True:
        start_number = len(lhs)

        # Item pass
        item_counts = lhs.item.value_counts()
        item_above = set(item_counts[item_counts >= k].index)
        lhs = lhs[lhs.item.isin(item_above)]
        print('Records after item pass: ', len(lhs))

        # User pass
        user_counts = lhs.user.value_counts()
        user_above = set(user_counts[user_counts >= k].index)
        lhs = lhs[lhs.user.isin(user_above)]
        print('Records after user pass: ', len(lhs))

        if len(lhs) == start_number:
            print('Exiting...')
            break
    return lhs


def create_index(lhs: pd.DataFrame):
    """
    Associate an index for each user and item after having performed filtering steps. In order to avoid confusion, it
    sorts the data by timestamp, user, and item before creating the indexes.
    @param lhs: Pandas Dataframe containing the listening records. Has columns ["timestamp", "user", "item"]
    @return:
        lhs: Pandas Dataframe containing the listening records, now with the user_idx and item_idx columns
        user_idxs: Pandas Dataframe containing the user to user_idx mapping
        item_idxs: Pandas Dataframe containing the item to item_idx mapping
    """
    # Defining a unique order for the index assignment
    lhs = lhs.sort_values(['timestamp', 'user', 'item'])

    # Creating simple integer indexes used for sparse matrices
    user_idxs = lhs.user.drop_duplicates().reset_index(drop=True)
    item_idxs = lhs.item.drop_duplicates().reset_index(drop=True)
    user_idxs.index.name = 'user_idx'
    item_idxs.index.name = 'item_idx'
    user_idxs = user_idxs.reset_index()
    item_idxs = item_idxs.reset_index()
    lhs = lhs.merge(user_idxs).merge(item_idxs)
    return lhs, user_idxs, item_idxs


def save_index(result_dir, lhs, user_indices, item_indices):
    lhs.to_csv(os.path.join(result_dir, 'listening_history.csv'), index=False)
    user_indices.to_csv(os.path.join(result_dir, 'user_idxs.csv'), index=False)
    item_indices.to_csv(os.path.join(result_dir, 'item_idxs.csv'), index=False)


def split_temporal_order_ratio_based(lhs: pd.DataFrame, ratios=(0.8, 0.1, 0.1)):
    """
    Split the interation time-wise, for each user, using the ratio specified as parameters. E.g. A split with (0.7,
    0.2,0.1) will first sort the data by timestamp then take the first 70% of the interactions as train data,
    the subsequent 20% as validation data, and the remaining last 10% as test data. In order to avoid confusion,
    it sorts the data by timestamp, user_idx, and item_idx before splitting.
    @param lhs: lhs: Pandas Dataframe
    containing the listening records. Has columns ["timestamp", "user_idx", "item_idx"]
    @param ratios: float values that denote the ratios for train, val, test. The values must sum to 1.
    @return:
        lhs: Pandas Dataframe containing the listening records sorted.
        train_data: Pandas Dataframe containing the train data.
        val_data: Pandas Dataframe containing the val data.
        test_data: Pandas Dataframe containing the test data.

    """
    assert sum(ratios) == 1, 'Ratios do not sum to 1!'

    lhs = lhs.sort_values(['timestamp', 'user', 'item'])
    train_idxs = []
    val_idxs = []
    test_idxs = []
    for user, user_group in tqdm(lhs.groupby('user')):
        # Data is already sorted by timestamp
        n_test = math.ceil(len(user_group) * ratios[-1])
        n_val = math.ceil(len(user_group) * ratios[-2])
        n_train = len(user_group) - n_val - n_test

        train_idxs += list(user_group.index[:n_train])
        val_idxs += list(user_group.index[n_train:n_train + n_val])
        test_idxs += list(user_group.index[-n_test:])

    train_data = lhs.loc[train_idxs]
    val_data = lhs.loc[val_idxs]
    test_data = lhs.loc[test_idxs]

    return lhs, train_data, val_data, test_data


def split_random_order_ratio_based(lhs: pd.DataFrame, ratios=(0.8, 0.1, 0.1), seed=13):
    """
    Split the interation time-wise, for each user, using the ratio specified in the configuration. As an example,
    a split with (0.7, 0.2, 0.1) will first randomize the data then take the first 70% of the interactions as
    train data, the subsequent 20% as validation data, and the remaining last 10% as test data.
    @param lhs: lhs: Pandas Dataframe
    containing the listening records. Has columns ["timestamp", "user_idx", "item_idx"]
    @param ratios: float values that denote the ratios for train, val, test. The values must sum to 1.
    @param seed: seed for the randomization
    @return:
        lhs: Pandas Dataframe containing the listening records sorted.
        train_data: Pandas Dataframe containing the train data.
        val_data: Pandas Dataframe containing the val data.
        test_data: Pandas Dataframe containing the test data.

    """
    assert sum(ratios) == 1, 'Ratios do not sum to 1!'
    lhs = lhs.sample(frac=1., random_state=seed)
    train_idxs = []
    val_idxs = []
    test_idxs = []
    for user, user_group in tqdm(lhs.groupby('user')):
        n_test = math.ceil(len(user_group) * ratios[-1])
        n_val = math.ceil(len(user_group) * ratios[-2])
        n_train = len(user_group) - n_val - n_test

        train_idxs += list(user_group.index[:n_train])
        val_idxs += list(user_group.index[n_train:n_train + n_val])
        test_idxs += list(user_group.index[-n_test:])

    train_data = lhs.loc[train_idxs]
    val_data = lhs.loc[val_idxs]
    test_data = lhs.loc[test_idxs]

    return lhs, train_data, val_data, test_data


def split_ratio(a, ratios):
    n_samples = len(a)
    n_val = math.ceil(n_samples * ratios[1])
    n_test = math.ceil(n_samples * ratios[2])
    n_train = n_samples - n_val - n_val
    return a[:n_train], a[n_train: n_train + n_val], a[-n_test:]


def split_temporal_order_based(data: RawDataset, config: SplitConfig, verbose: bool = True) -> AllSplitsData:
    """
    Split the interation time-wise, for each user, using the ratio specified in the configuration. As an example,
    a split with (0.7, 0.2, 0.1) will first sort the data by timestamp then take the first 70% of the interactions
    as train data, the subsequent 20% as validation data, and the remaining last 10% as test data. In order to avoid
    confusion, it sorts the data by timestamp, user_idx, and item_idx before splitting.
    @param data: the data to split
    @param config: configuration to use for splitting
    @param verbose: whether to write information on progress to stdout
    @return: an 'AllSplitsData' instance containing the data for the different splits
    """
    print_verbose(f'performing temporal split with ratios {config.ratios}', verbose=verbose)

    if sum(config.ratios) != 1:
        raise ValueError('ratios do not sum up to 1')

    tr_indices, vd_indices, te_indices = [], [], []
    lhs = data.interactions.sort_values(by='timestamp')
    for user, user_group in tqdm(lhs.groupby('user')):
        tr, vd, te = split_ratio(user_group.index, config.ratios)
        tr_indices += list(tr)
        vd_indices += list(vd)
        te_indices += list(te)

    split_results = {}
    # user and item indices as well as their features are the same across the split
    user_indices = np.array(sorted(lhs['user_idx'].unique()))
    item_indices = np.array(sorted(lhs['item_idx'].unique()))
    for split, split_indices in zip(SPLIT_NAMES, [tr_indices, vd_indices, te_indices]):
        print_verbose(f'performing "{split}" split')
        split_history = lhs.loc[split_indices]

        split_results[split] = SplitData(
            interactions=split_history,
            user_indices=user_indices,
            item_indices=item_indices,
            user_features=data.user_features,
            item_features=data.item_features
        )

        print_description_listening_history(split_history, epilogue=f'in temporal "{split}" split',
                                            verbose=verbose)

    return AllSplitsData(
        tr_data=split_results['train'],
        vd_data=split_results['val'],
        te_data=split_results['test'],
    )


def split_random_order_based(data: RawDataset, config: SplitConfig, verbose: bool = True) -> AllSplitsData:
    """
    Split the interaction in a random fashion, for each user, using the ratio defined as . E.g. A split with (0.7,
    0.2,0.1) will first randomize the data then take the first 70% of the interactions as train data,
    the subsequent 20% as validation data, and the remaining last 10% as test data.
    @param data: the data to split
    @param config: configuration to use for splitting
    @param verbose: whether to write information on progress to stdout
    @return: an 'AllSplitsData' instance containing the data for the different splits
    """
    print_verbose(f'performing random split with ratios {config.ratios}', verbose=verbose)

    if sum(config.ratios) != 1:
        raise ValueError('ratios do not sum up to 1')

    tr_indices, vd_indices, te_indices = [], [], []

    # randomly sort data for splitting
    lhs = data.interactions.sample(frac=1., random_state=config.seed)
    for user, user_group in tqdm(lhs.groupby('user')):
        tr, vd, te = split_ratio(user_group.index, config.ratios)
        tr_indices += list(tr)
        vd_indices += list(vd)
        te_indices += list(te)

    split_results = {}
    # user and item indices as well as their features are the same across the split
    user_indices = np.array(sorted(lhs['user_idx'].unique()))
    item_indices = np.array(sorted(lhs['item_idx'].unique()))
    for split, split_indices in zip(SPLIT_NAMES, [tr_indices, vd_indices, te_indices]):
        print_verbose(f'performing "{split}" split', verbose=verbose)
        split_history = lhs.loc[split_indices]

        split_results[split] = SplitData(
            interactions=split_history,
            user_indices=user_indices,
            item_indices=item_indices,
            user_features=data.user_features,
            item_features=data.item_features
        )

        print_description_listening_history(split_history, epilogue=f'in random "{split}" split',
                                            verbose=verbose)

    return AllSplitsData(
        tr_data=split_results['train'],
        vd_data=split_results['val'],
        te_data=split_results['test'],
    )


def split_cold_start_ratio_based(data: RawDataset, config: SplitConfig, verbose: bool = True) -> AllSplitsData:
    """
    Splits users and items into train, validation and test set based on the ColdStartType in the config.
    See below for schematics, we consider only the user-item interaction matrix for simplicity.

          USER COLD START                    ITEM COLD START                       USER ITEM COLD START
    +-------+-----------------+   +-------+----------------------------+   +-------+---------------------------+
    |       |      items      |   |       |            items           |   |       |           items           |
    +-------+-----------------+   +-------+---------------+-----+------+   +-------+-------+------------+------+
    |       |                 |   |       |               |     |      |   |       |              |     |      |
    |       |      train      |   | users |     train     | val | test |   |       |              |     |      |
    |       |                 |   |       |               |     |      |   |       |    train     |     |      |
    | users +-----------------+   +-------+---------------+-----+------+   |       |              |     |      |
    |       |    validation   |                                            | users |              |     |      |
    |       +-----------------+                                            |       +--------------+     |      |
    |       |      test       |                                            |       |                val |      |
    +-------+-----------------+                                            |       +--------------------+      |
                                                                           |       |                      test |
                                                                           +-------+-------+------------+------+

    @param data: the dataset to split
    @param config: configuration to use for splitting
    @param verbose: whether to write information on progress to stdout
    @return: an 'AllSplitsData' instance containing the data for the different splits
    """
    # extract to variables for usage brevity
    seed = config.seed
    ratios = config.ratios
    cold_start_scenario = config.cold_start_type

    print_verbose(f'Performing {cold_start_scenario} split with ratios {ratios}', verbose=verbose)

    if sum(ratios) != 1:
        raise ValueError('Ratios do not sum up to 1')

    # use random number generator for reproducibility
    rng = np.random.default_rng(seed)

    # retrieve unique users, sorted to ensure determinism
    unique_user_indices = sorted(list(data.interactions['user_idx'].unique()))
    if cold_start_scenario in [ColdStartType.User, ColdStartType.Both]:
        rng.shuffle(unique_user_indices)
        user_split_indices = split_ratio(unique_user_indices, ratios)
    else:
        # use all users in every split
        user_split_indices = (unique_user_indices,) * 3

    # retrieve unique items, sorted to ensure determinism
    unique_item_indices = sorted(list(data.interactions['item_idx'].unique()))
    if cold_start_scenario in [ColdStartType.Item, ColdStartType.Both]:
        rng.shuffle(unique_item_indices)
        item_split_indices = split_ratio(unique_item_indices, ratios)
    else:
        # use all items in every split
        item_split_indices = (unique_item_indices,) * 3

    split_results = {}
    for split, user_indices, item_indices in zip(SPLIT_NAMES, user_split_indices, item_split_indices):
        print_verbose(f'performing "{split}" split')
        split_history = data.interactions[data.interactions['user_idx'].isin(user_indices) &
                                          data.interactions['item_idx'].isin(item_indices)]

        split_user_features = filter_based_on_indices('user_idx', data.user_features, user_indices)
        split_item_features = filter_based_on_indices('item_idx', data.item_features, item_indices)
        sort_based_on_indices('user', data.user_features)
        sort_based_on_indices('item', data.item_features)

        split_results[split] = SplitData(
            interactions=split_history,
            user_indices=user_indices,
            item_indices=item_indices,
            user_features=split_user_features,
            item_features=split_item_features
        )

        print_description_listening_history(split_history, epilogue=f'in "{split}" split and '
                                                                    f'"{cold_start_scenario}" cold-start scenario ',
                                            verbose=verbose)

    return AllSplitsData(
        tr_data=split_results['train'],
        vd_data=split_results['val'],
        te_data=split_results['test'],
    )


def split_ratio_based(data: RawDataset, config: SplitConfig, verbose: bool = True) -> AllSplitsData:
    """
    Splits the dataset based on the specified config
    @param data: the dataset to split
    @param config: configuration to use for splitting
    @param verbose: whether to write information on progress to stdout
    """
    match config.split_type:
        case SplitType.Temporal:
            return split_temporal_order_based(data, config, verbose)
        case SplitType.ColdStart:
            return split_cold_start_ratio_based(data, config, verbose)
        case SplitType.Random:
            return split_random_order_based(data, config, verbose)
        case _:
            raise ValueError(f'Split type {config.split_type} is not supported.')


def get_default_split_path(base_path: str, config: SplitConfig):
    name_map = {
        SplitType.Random: 'random_split',
        SplitType.Temporal: 'temporal_split',
        SplitType.ColdStart: f'cold_start_{config.cold_start_type}'
    }
    return os.path.join(base_path, name_map[config.split_type])


def store_feature_data(result_dir: str, entity: str, data: EntityFeatures, filename_postfix: str = ''):
    if data.tabular_features is not None:
        feature_file = os.path.join(result_dir, f'{entity}_features{filename_postfix}.csv')
        data.tabular_features.to_csv(feature_file, index=False)
    if data.multidimensional_features is not None:
        for feature_name, feature_data in data.multidimensional_features.items():
            assert len(feature_data.indices) == len(feature_data.values), \
                'Feature filtering failed somewhere as number of feature indices don\'t match number of actual features'

            feature_file_path = os.path.join(result_dir, f'{entity}_{feature_name}{filename_postfix}.npz')
            np.savez(feature_file_path, indices=feature_data.indices, values=feature_data.values)


def store_data(result_dir: str, data: RawDataset, filename_postfix: str = ''):
    # store interactions
    interaction_path = os.path.join(result_dir, f'listening_history{filename_postfix}.csv')
    data.interactions.to_csv(interaction_path, index=False)

    store_feature_data(result_dir, 'user', data.user_features, filename_postfix)
    store_feature_data(result_dir, 'item', data.item_features, filename_postfix)


def store_split(result_dir: str, split: str, data: SplitData):
    store_data(result_dir, data, filename_postfix=f'_{split}')


def store_splits(result_dir: str, data: AllSplitsData):
    store_split(result_dir, 'train', data.tr_data)
    store_split(result_dir, 'val', data.vd_data)
    store_split(result_dir, 'test', data.te_data)


def load_features(data_dir: str, entity: str, feature_configs: List[FeatureBaseDefinition],
                  filename_postfix: str = '') -> EntityFeatures:
    multi_d_feature_names = [f.name for f in feature_configs if f.type in [FeatureType.VECTOR, FeatureType.MATRIX]]
    tabular_feature_names = [f.name for f in feature_configs if f.name not in multi_d_feature_names]

    index_columns = [entity, f'{entity}_idx']
    table_columns_to_load = index_columns + tabular_feature_names

    feature_file = os.path.join(data_dir, f'{entity}_features{filename_postfix}.csv')
    tabular_features = None
    if len(tabular_feature_names) > 0:
        if os.path.exists(feature_file):
            # only load specific columns for an "improved parsing time and lower memory usage"
            # (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
            # using 'lambda' prevents a crash if the '{entity}_idx' column does not exist
            tabular_features = pd.read_csv(feature_file, usecols=lambda x: x in table_columns_to_load)
            loaded_feature_columns = set(tabular_features.columns).difference(index_columns)
            missing_feature_columns = set(tabular_feature_names).difference(loaded_feature_columns)
            if len(missing_feature_columns) > 0:
                raise ValueError(f'Column(s) for {entity} feature(s) {list(missing_feature_columns)} are missing.')
        else:
            raise FileNotFoundError(f'Feature file "{feature_file}" does not exist')

    multi_d_features = {}
    for fc in feature_configs:
        if fc.name in multi_d_feature_names:
            feature_file = os.path.join(data_dir, f'{entity}_{fc.name}{filename_postfix}.npz')
            if not os.path.exists(feature_file):
                raise FileNotFoundError(f'Data file for {entity} feature "{fc.name}" does not exist.')
            loaded_data = np.load(feature_file, allow_pickle=True)
            feature_indices = loaded_data['indices']
            feature_values = loaded_data['values']

            if len(feature_indices) != len(feature_values):
                raise ValueError(f'Mismatch between number of {entity} indices and its "{fc.name}" feature'
                                 f'({len(feature_indices)} indices but {len(feature_values)} feature values).')

            multi_d_features[fc.name] = MultiDFeature(feature_indices, feature_values)

    return EntityFeatures(tabular_feature_names, tabular_features, multi_d_features)


def load_split_features(data_dir: str, entity: str, feature_configs: List[FeatureBaseDefinition],
                        split: str) -> EntityFeatures:
    return load_features(data_dir, entity, feature_configs, f'_{split}')


def merge_features(entity: str, feature_entities: list[EntityFeatures]) -> EntityFeatures:
    all_features = None
    for feature_entity in feature_entities:
        if all_features is None:
            all_features = feature_entity
            continue
        else:
            if set(all_features.tabular_feature_names) != set(feature_entity.tabular_feature_names):
                raise ValueError('Tabular feature names of the different splits must be identical')

            if all_features.tabular_features is not None and feature_entity.tabular_features is not None:
                column = f'{entity}_idx'
                new_data_mask = ~feature_entity.tabular_features[column].isin(all_features.tabular_features[column])
                all_features.tabular_features = pd.concat([all_features.tabular_features,
                                                           feature_entity.tabular_features[new_data_mask]])

            if set(all_features.multidimensional_features.keys()) != \
                    set(feature_entity.multidimensional_features.keys()):
                raise ValueError('Multidimensional feature names of the different splits must be identical')

            for k in all_features.multidimensional_features.keys():
                all_indices = all_features.multidimensional_features[k].indices
                all_values = all_features.multidimensional_features[k].values

                split_indices = feature_entity.multidimensional_features[k].indices
                split_values = feature_entity.multidimensional_features[k].values

                new_data_mask = np.isin(split_indices, all_indices, assume_unique=True, invert=True)

                # determine new indices and values
                new_indices = np.concatenate([all_indices, split_indices[new_data_mask]], axis=0)
                new_values = np.concatenate([all_values, split_values[new_data_mask]], axis=0)

                # ... and store them
                all_features.multidimensional_features[k].indices = new_indices
                all_features.multidimensional_features[k].values = new_values

    if all_features is not None:
        sort_based_on_indices(entity, all_features, verbose=False)
    return all_features


def load_all_features(data_dir: str, entity: str, feature_configs: List[FeatureBaseDefinition],
                      splits: tuple[str] = SPLIT_NAMES) -> EntityFeatures:
    feature_entities = [load_split_features(data_dir, entity, feature_configs, split) for split in splits]
    all_features = merge_features(entity, feature_entities)
    return all_features


def load_data(data_dir: str, config: DataPreprocessingConfig, verbose: bool = True) -> RawDataset:
    print_verbose(f'loading interaction and feature data from "{data_dir}"', verbose=True)
    interaction_path = os.path.join(data_dir, 'listening_history.csv')
    interaction_history = pd.read_csv(interaction_path)

    return RawDataset(
        interactions=interaction_history,
        user_features=load_features(data_dir, 'user', config.user_features),
        item_features=load_features(data_dir, 'item', config.item_features),
    )
