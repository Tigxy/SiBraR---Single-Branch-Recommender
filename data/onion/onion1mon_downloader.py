import argparse
import os
import shutil
from dask import dataframe as dd
import pandas as pd

from data.data_preprocessing_utils import store_data
from data.data_download_utils import download_onion_dataset
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
parser.add_argument('--zenodo_access_token', '-zat', help='Zenodo Access Token')
parser.add_argument('--config_file', '-c', help='Configuration file defining the preprocessing')
parser.add_argument('--save_path', '-s', help='The path where to store the dataset', default='./')
parser.add_argument('--year', help='Which year the listening history should be taken from.',
                    required=False, default=None)
parser.add_argument('--month', help='Which month the listening history should be taken from.',
                    required=False, default=None)
args = parser.parse_args()

# get general parameters
force_download = args.force_download
zenodo_access_token = args.zenodo_access_token
config_file = args.config_file
save_path = args.save_path

subset_year = args.year
subset_month = args.month

lhs_subset_filename = f'userid_trackid_timestamp_{subset_month}_{subset_year}.tsv'

with open(config_file) as fh:
    config = DataPreprocessingConfig.from_yaml(fh)

user_feature_names = [user_feature.name for user_feature in config.user_features]
item_feature_names = [item_feature.name for item_feature in config.item_features]

raw_dataset_path = os.path.join(save_path, 'raw_dataset')

# clean up previously processed dataset
processed_dataset_path = os.path.join(save_path, 'processed_dataset')
if os.path.exists(processed_dataset_path):
    shutil.rmtree(processed_dataset_path)
os.makedirs(processed_dataset_path, exist_ok=False)

listening_history_subset_path = os.path.join(raw_dataset_path, lhs_subset_filename)
subset_exists = os.path.exists(listening_history_subset_path)

log_file = os.path.join(processed_dataset_path, 'onion_downloader.log')
with Tee(log_file) as tee:
    # download dataset
    download_onion_dataset(
        zenodo_access_token,
        item_feature_names,
        save_path,
        force_download,
        do_not_download_listening_history=subset_exists
    )

    users_data_path = os.path.join(raw_dataset_path, 'users.tsv')
    listening_events_path = os.path.join(raw_dataset_path, 'userid_trackid_timestamp.tsv')
    features_folder_path = os.path.join(raw_dataset_path, 'features')

    # get paths for the item features
    feature_file_map = {feature_name: os.path.join(features_folder_path, f"id_{feature_name}.tsv")
                        for feature_name in item_feature_names}

    # Loading users
    users = pd.read_csv(users_data_path,
                        delimiter='\t',
                        usecols=['user_id'] + user_feature_names
                        ).rename(columns={'user_id': 'user'})

    print("loading listening history...")
    if subset_exists:
        print("dataset snippet already exists, reusing it")
        lhs = pd.read_csv(listening_history_subset_path, sep='\t', names=['user', 'item', 'timestamp'], header=0)
    else:
        USE_PD = True
        df_lib = pd if USE_PD else dd
        read_params = {'engine': 'pyarrow'} if USE_PD else {}

        # Keeping the data only from the last month
        lhs = df_lib.read_csv(listening_events_path, sep='\t', names=['user', 'item', 'timestamp'], header=0,
                              **read_params)

        if subset_year is not None:
            lhs = lhs[df_lib.to_datetime(lhs.timestamp).dt.year == subset_year]
        if subset_month is not None:
            lhs = lhs[df_lib.to_datetime(lhs.timestamp).dt.month == subset_month]

        if not USE_PD:
            lhs = lhs.compute()

        lhs.to_csv(listening_history_subset_path, sep='\t', index=False)
    print("listening history loaded")

    print_description_listening_history(lhs)

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
            tabular_features=users
        ),
        item_features=EntityFeatures(
            multidimensional_features=item_multi_d_features
        ),
    )
    store_data(processed_dataset_path, data)

    print(f'All files processed, bye!')
