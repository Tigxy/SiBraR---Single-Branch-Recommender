import argparse
import os
import shutil

import pandas as pd
from dask import dataframe as dd

from data.data_preprocessing_utils import LOG_FILT_DATA_PATH, print_and_log, k_core_filtering, create_index, \
    split_temporal_order_ratio_based
from data.data_download_utils import download_amazon2024_dataset
from data.amazon2024.filter_on_meta import filter_on_meta, filter_on_verified
import pickle
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--force_download', '-d', action='store_true',
                    help='Whether or not to re-download the dataset if "raw_dataset" folder is detected. Default to '
                         'False',
                    default=False)

parser.add_argument('--max_year', '-maxy', help='The max year (<=) to restrict to interactions to. Default is None',
                    default=None)

parser.add_argument('--min_year', '-miny', help='The min year (>) to restrict to interactions to. Default is None',
                    default=None)

parser.add_argument('--category', '-c',
                    help='Which category of the amazon2024 dataset to preprocess. Default is videogames',
                    type=str,
                    default='videogames',
                    )

parser.add_argument('--threshold', '-t',
                    help='Threshold to set on the rating for binarization. Default is -1 (keep all)',
                    type=float,
                    default=-1.,
                    )

parser.add_argument('--k_core', '-k',
                    help='k for core filtering. Default is 1 (keep all)',
                    type=int,
                    default=10,
                    )

parser.add_argument('--verified_true', '-v',
                    help='Filter on verified purchases. Default is False', action='store_true',
                    default=True,
                    )

categories_short_to_official = {
    'beauty': 'All_Beauty',
    'software': 'Software',
    'clothing': 'Clothing_Shoes_and_Jewelry',
    'videogames': 'Video_Games',
    'fashion': 'Amazon_Fashion',
}

large_ds = ['clothing']
args = parser.parse_args()
force_download = args.force_download
category_short = args.category
max_year = args.max_year
min_year = args.min_year
threshold = args.threshold
k_core = args.k_core
verified_true = args.verified_true

category_official = categories_short_to_official[category_short]

df_lib = dd if category_short in large_ds else pd

print(f"Processing {category_short}...")
if not os.path.exists(f'./{category_short}/raw_dataset') or force_download:
    if force_download and os.path.exists(f'./{category_short}/raw_dataset'):
        shutil.rmtree(f'./{category_short}/raw_dataset')
    download_amazon2024_dataset('./', category_official=category_official, category_short=category_short)

if os.path.exists(f'./{category_short}/processed_dataset'):
    shutil.rmtree(f'./{category_short}/processed_dataset')
os.mkdir(f'./{category_short}/processed_dataset')

ratings_path = f'./{category_short}/raw_dataset/{category_official}_verified.csv' if verified_true else f'./{category_short}/raw_dataset/{category_official}.csv'
log_filt_data_file = open(os.path.join(f'./{category_short}/processed_dataset', LOG_FILT_DATA_PATH), 'w+')

lhs = df_lib.read_csv(ratings_path, names=['user', 'item', 'rating', 'timestamp'], skiprows=[0])

if category_short not in large_ds:
    (log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(), 'Original Data')
# keeping only one year
if max_year:
    lhs = lhs[lhs.timestamp.apply(lambda x: datetime.datetime.fromtimestamp(x / 1e3).year <= max_year)]
    if category_short in large_ds:
        lhs = lhs.compute()
        print(f'Only ts <= {max_year}')
    else:
        print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(), f'Only ts <= {max_year}')

if min_year:
    lhs = lhs[lhs.timestamp.apply(lambda x: datetime.datetime.fromtimestamp(x / 1e3).year > min_year)]
    if category_short in large_ds:
        lhs = lhs.compute()
        print(f'Only ts > {min_year}')
    else:
        print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(), f'Only ts > {min_year}')

# keeping only all items with selected features
filtered_items = filter_on_meta(category_official, category_short, crawl_images=False)
lhs = lhs[lhs.item.isin(filtered_items)]

# Keeping only the first interaction
lhs = lhs.sort_values('timestamp')
lhs = lhs.drop_duplicates(subset=['item', 'user'])

if category_short not in large_ds:
    print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(), 'Only first interaction')
else:
    print('Only first interaction')

# We keep all ratings
lhs = lhs[lhs.rating >= threshold]
if category_short in large_ds and not year_2018:
    lhs = lhs.compute()

print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(),
              'Only Positive Interactions (>= -1.)')

lhs = k_core_filtering(lhs, k_core)

print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(),
              '5-core filtering')
lhs.to_csv(f'./{category_short}/processed_dataset/listening_history.csv', index=False)
"""
lhs, user_idxs, item_idxs = create_index(lhs)

print('Splitting the data, temporal ordered - ratio-based (80-10-10)')

lhs, train_data, val_data, test_data = split_temporal_order_ratio_based(lhs)

print_and_log(log_filt_data_file, len(train_data), train_data.user.nunique(), train_data.item.nunique(), 'Train Data')
print_and_log(log_filt_data_file, len(val_data), val_data.user.nunique(), val_data.item.nunique(), 'Val Data')
print_and_log(log_filt_data_file, len(test_data), test_data.user.nunique(), test_data.item.nunique(), 'Test Data')

log_filt_data_file.close()

# Saving locally
print('Saving data to ./processed_dataset')

lhs.to_csv(f'./{category_short}/processed_dataset/listening_history.csv', index=False)
train_data.to_csv(f'./{category_short}/processed_dataset/listening_history_train.csv', index=False)
val_data.to_csv(f'./{category_short}/processed_dataset/listening_history_val.csv', index=False)
test_data.to_csv(f'./{category_short}/processed_dataset/listening_history_test.csv', index=False)

user_idxs.to_csv(f'./{category_short}/processed_dataset/user_idxs.csv', index=False)
item_idxs.to_csv(f'./{category_short}/processed_dataset/item_idxs.csv', index=False)
"""
