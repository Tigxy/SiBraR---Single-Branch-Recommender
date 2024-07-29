import os
import yaml
import shutil
import argparse

from data.data_preprocessing_utils import store_splits, load_data, \
    create_index, save_index, split_ratio_based, get_default_split_path
from data.feature_normalization import normalize_features
from data.filtering import filter_interactions, filter_unique, filter_k_core, \
    print_description_listening_history, filter_tabular_features, filter_history, \
    filter_based_on_history, filter_entities_without_all_features, \
    update_indices, sort_based_on_indices
from data.preprocessing_config_classes import DataPreprocessingConfig

# get arguments from command line
from data.tee import Tee

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', '-c', help='.yaml configuration file defining the preprocessing',
                    required=True)
parser.add_argument('--data_path', '-d', help='The path where the data is stored', default='./')
parser.add_argument('--split_path', '-s', required=False,
                    help='The path where to store the split data to. '
                         'If not specified, it will default to {data_path}/{split_config}', default=None)
args = parser.parse_args()

config_file = args.config_file
data_path = args.data_path

# load configuration file based on which we perform the preprocessing
with open(config_file) as fh:
    config_dict = yaml.safe_load(fh)
    config = DataPreprocessingConfig.from_dict(config_dict)

user_feature_names = [user_feature.name for user_feature in config.user_features]
item_feature_names = [item_feature.name for item_feature in config.item_features]

# clean up previously split dataset
split_path = args.split_path or get_default_split_path(data_path, config.split)
if os.path.exists(split_path):
    shutil.rmtree(split_path)
os.makedirs(split_path, exist_ok=False)

log_file = os.path.join(split_path, 'preprocessor.log')

with Tee(log_file) as tee:
    print('copy config file to results directory')
    shutil.copyfile(config_file, os.path.join(split_path, 'used_config.yaml'))

    # load data from dataset
    data = load_data(data_path, config)
    lhs = data.interactions
    print_description_listening_history(lhs, epilogue='in listening history')

    # drop features of users & items that are not in the listening history
    data.user_features = filter_based_on_history(lhs, entity='user', entity_features=data.user_features)
    data.item_features = filter_based_on_history(lhs, entity='item', entity_features=data.item_features)

    # for all loaded users and items we need all features
    data.user_features = filter_entities_without_all_features('user', data.user_features)
    data.item_features = filter_entities_without_all_features('item', data.item_features)

    # filter features based on the configuration
    data.user_features.tabular_features = filter_tabular_features('user', data.user_features.tabular_features,
                                                                  config.user_features)
    data.item_features.tabular_features = filter_tabular_features('item', data.item_features.tabular_features,
                                                                  config.item_features)

    # drop histories for previously filtered users & items
    lhs = filter_history(lhs, 'user', data.user_features)
    lhs = filter_history(lhs, 'item', data.item_features)

    # filter the listening history
    lhs = filter_interactions(lhs, config.interactions.min_n_interactions)
    lhs = filter_unique(lhs)
    lhs = filter_k_core(lhs, config.interactions.k_core)

    # drop features of previously filtered users & items
    data.user_features = filter_based_on_history(lhs, entity='user', entity_features=data.user_features)
    data.item_features = filter_based_on_history(lhs, entity='item', entity_features=data.item_features)

    # as we've reached the final number of users and items, we reset their indices
    lhs, user_indices_map, item_indices_map = create_index(lhs)
    save_index(split_path, lhs, user_indices_map, item_indices_map)
    data.interactions = lhs

    # update feature indices to match their new values
    data.user_features = update_indices('user', data.user_features, user_indices_map)
    data.item_features = update_indices('item', data.item_features, item_indices_map)

    # restore order items
    sort_based_on_indices('user', data.user_features)
    sort_based_on_indices('item', data.item_features)

    split_results = split_ratio_based(data, config=config.split)

    # perform final normalization (doing this after splitting to ensure no leakage from train to validation data)
    split_results = normalize_features(split_results, config, verbose=True)

    store_splits(split_path, split_results)
    print(f'all files processed, bye!')
