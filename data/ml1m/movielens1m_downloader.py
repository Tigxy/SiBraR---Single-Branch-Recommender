import argparse
import os
import shutil
from dask import dataframe as dd
import pandas as pd

from data.data_preprocessing_utils import store_data
from data.data_download_utils import download_onion_dataset, download_movielens_dataset
from data.data_utils import split_feature_id_from_data
from data.filtering import print_description_listening_history
from data.preprocessing_config_classes import DataPreprocessingConfig
from data.preprocessing_data_classes import RawDataset, EntityFeatures, MultiDFeature

# get arguments from command line
from data.tee import Tee

parser = argparse.ArgumentParser()

# general parameters
parser.add_argument('--force_download', '-d', action='store_true',
                    help='Whether or not to re-download the dataset if "raw_dataset" folder is detected. Default to '
                         'False')
parser.add_argument('--config_file', '-c', help='Configuration file defining the preprocessing')
parser.add_argument('--save_path', '-s', help='The path where to store the dataset', default='./')
args = parser.parse_args()

# get general parameters
force_download = args.force_download
config_file = args.config_file
save_path = args.save_path

with open(config_file) as fh:
    config = DataPreprocessingConfig.from_yaml(fh)

user_feature_names = [user_feature.name for user_feature in config.user_features]
item_feature_names = [item_feature.name for item_feature in config.item_features]

# clean up previously processed dataset
processed_dataset_path = os.path.join(save_path, 'processed_dataset')
if os.path.exists(processed_dataset_path):
    shutil.rmtree(processed_dataset_path)
os.makedirs(processed_dataset_path, exist_ok=False)

log_file = os.path.join(processed_dataset_path, 'ml1m_downloader.log')
with Tee(log_file) as tee:
    # download dataset
    raw_dataset_path = os.path.join(save_path, 'raw_dataset')
    if not os.path.exists(raw_dataset_path) or force_download:
        if force_download and os.path.exists(raw_dataset_path):
            shutil.rmtree(raw_dataset_path)
        download_movielens_dataset(raw_dataset_path, '1m')

    users_data_path = os.path.join(raw_dataset_path, 'users.dat')
    items_data_path = os.path.join(raw_dataset_path, 'movies.dat')
    listening_events_path = os.path.join(raw_dataset_path, 'ratings.dat')
    features_folder_path = os.path.join(raw_dataset_path, 'features')

    # get paths for the item features
    feature_file_map = {feature_name: os.path.join(features_folder_path, f"id_{feature_name}.tsv")
                        for feature_name in item_feature_names}

    # Loading users
    users = pd.read_csv(users_data_path, sep='::', index_col=False, encoding='latin-1', engine='python',
                        names=['user', 'gender', 'age', 'occupation', 'zip-code'])

    # for later text encoding, it might be nice to have the actual names of the occupation
    occupation_dict = {
        0: "other or not specified",
        1: "academic/educator",
        2: "artist",
        3: "clerical/admin",
        4: "college/grad student",
        5: "customer service",
        6: "doctor/health care",
        7: "executive/managerial",
        8: "farmer",
        9: "homemaker",
        10: "K-12 student",
        11: "lawyer",
        12: "programmer",
        13: "retired",
        14: "sales/marketing",
        15: "scientist",
        16: "self-employed",
        17: "technician/engineer",
        18: "tradesman/craftsman",
        19: "unemployed",
        20: "writer"
    }
    users = users.assign(occupation_str=users['occupation'].apply(lambda idx: occupation_dict[idx]))

    items = pd.read_csv(items_data_path, sep='::', index_col=False, encoding='latin-1', engine='python',
                        names=['item', 'title', 'genres'])

    lhs = pd.read_csv(listening_events_path, sep='::', index_col=False, encoding='latin-1', engine='python',
                      names=['user', 'item', 'rating', 'timestamp'])

    print_description_listening_history(lhs, epilogue='in full listening history')

    lhs = lhs[lhs['rating'] >= 3]
    print_description_listening_history(lhs, epilogue='in history with ratings >= 3')

    user_multi_d_features = {}
    item_multi_d_features = {}
    for feature_name, feature_file in feature_file_map.items():
        # load the features
        feature_df = pd.read_csv(feature_file, sep='\t')

        # rename just in case they were upper-cased
        column_renames = {'ID': 'id'}
        feature_df = feature_df.rename(columns=column_renames)

        feature_indices, feature_values = split_feature_id_from_data(feature_df, 'id')

        md_feature = MultiDFeature(feature_indices, feature_values)
        item_multi_d_features[feature_name] = md_feature

    data = RawDataset(
        interactions=lhs,
        user_features=EntityFeatures(
            tabular_features=users,
            multidimensional_features=user_multi_d_features
        ),
        item_features=EntityFeatures(
            tabular_features=items,
            multidimensional_features=item_multi_d_features
        ),
    )
    store_data(processed_dataset_path, data)

    print(f'All files processed, bye!')
