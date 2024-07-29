import bz2
import glob
import os
import shutil
import zipfile
from typing import Union

import requests
import gdown
import zenodopy

import gzip

MOVIELENS_100K_DATASET_LINK = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_1M_DATASET_LINK = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_10M_DATASET_LINK = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
LFM2B_2020_INTER_DATASET_LINK = "http://www.cp.jku.at/datasets/LFM-2b/recsys22/listening_events.tsv.bz2"
LFM2B_2020_USER_DATASET_LINK = "http://www.cp.jku.at/datasets/LFM-2b/recsys22/users.tsv.bz2"
LFM2B_2020_TRACK_DATASET_LINK = "http://www.cp.jku.at/datasets/LFM-2b/recsys22/tracks.tsv.bz2"
ONION_ZENODO_BUCKET = "https://zenodo.org/records/6609677/files"
AMAZONVID2018_DATASET_LINK = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games.csv"
AMAZON2024_LINK = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/benchmark/0core/rating_only/"
AMAZON2024_META_LINK = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/"
AMAZON2024_REVIEW_LINK = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/"
DELIVERY_HERO_SINGAPORE_DATASET_FILE_ID = "1v-FfCbLtv02EpNpopDx25EQnHZeT1nL2"
KUAIREC_DATASET_FILE_ID = "1qe5hOSBxzIuxBb1G_Ih5X-O65QElollE"


def download_movielens_dataset(save_path: str = './', which: str = '1m'):
    """
    Downloads a movielens dataset.

    @type save_path: path to the folder where to save the raw dataset. Default to "./"
    @param which: Which movielens dataset should be donwloaded.
    """
    assert which in ['100k', '1m',
                     '10m'], f'The current implementation manages only 1m and 10m! {which} is not valid value.'

    # Downloading
    if which == '100k':
        url = MOVIELENS_100K_DATASET_LINK
    elif which == '1m':
        url = MOVIELENS_1M_DATASET_LINK
    elif which == '10m':
        url = MOVIELENS_10M_DATASET_LINK
    else:
        raise ValueError(f'The current implementation manages only 100k, 1m and 10m! {which} is not valid value.')

    print("Downloading the dataset...")
    req = requests.get(url)
    dataset_zip_name = os.path.join(save_path, "dataset.zip")

    os.makedirs(save_path)
    with open(dataset_zip_name, 'wb') as fw:
        fw.write(req.content)

    # Unzipping
    with zipfile.ZipFile(dataset_zip_name, 'r') as zipr:
        zipr.extractall(save_path)

    os.remove(dataset_zip_name)

    downloaded_dir_map = {'100k': 'ml-100k', '1m': 'ml-1m', '10m': '10M100K'}
    downloaded_dir = os.path.join(save_path, downloaded_dir_map[which])
    for p in glob.glob(os.path.join(downloaded_dir, '*')):
        new_path = os.path.join(save_path, os.path.relpath(p, downloaded_dir))
        shutil.move(p, new_path)
    shutil.rmtree(downloaded_dir)

    print('Dataset downloaded')


def download_lfm2b_2020_dataset(save_path: str = './'):
    """
    Downloads the LFM2b 2020 Subset
    @type save_path: path to the folder where to save the raw dataset. Default to "./"
    """

    if not os.path.exists(os.path.join(save_path, 'raw_dataset')):
        os.makedirs(os.path.join(save_path, 'raw_dataset'))
    # Downloading
    print("Downloading the dataset...")
    print('Downloading interaction data...')
    req = requests.get(LFM2B_2020_INTER_DATASET_LINK)
    data = bz2.decompress(req.content)
    file_name = os.path.join(save_path, "raw_dataset", "inter_dataset.tsv")
    with open(file_name, 'wb') as fw:
        fw.write(data)

    print('Downloading user data...')
    req = requests.get(LFM2B_2020_USER_DATASET_LINK)
    data = bz2.decompress(req.content)
    file_name = os.path.join(save_path, "raw_dataset", "users.tsv")
    with open(file_name, 'wb') as fw:
        fw.write(data)

    print('Downloading track data...')
    req = requests.get(LFM2B_2020_TRACK_DATASET_LINK)
    data = bz2.decompress(req.content)
    file_name = os.path.join(save_path, "raw_dataset", "tracks.tsv")

    with open(file_name, 'wb') as fw:
        fw.write(data)


def soft_download_zenodo(zenodo_client: zenodopy.Client, remote_file: str,
                         local_path: str, force_download: bool = False):
    remote_dir, remote_file_name = os.path.split(remote_file)
    local_file = os.path.join(local_path, remote_dir, remote_file_name)
    if os.path.exists(local_file) and not force_download:
        print(f"file '{local_file}' already exists, skipping download")
        return
    zenodo_client.download_file(remote_file, local_path)


def soft_extract_bz2(source_file, target_file, override: bool = False):
    if os.path.exists(target_file) and not override:
        print(f"file '{target_file}' already exists, skipping extraction")
        return
    with open(target_file, 'wb') as decompressed_file, bz2.BZ2File(source_file, 'rb') as bz_file:
        for data in iter(lambda: bz_file.read(100 * 1024), b''):
            decompressed_file.write(data)


def soft_download_file(url: str, local_file: str, force_download: bool = False):
    if os.path.exists(local_file) and not force_download:
        print(f"file '{local_file}' already exists, skipping download")
        return
    req = requests.get(url)
    with open(local_file, 'wb') as fw:
        fw.write(req.content)


def soft_download_bz2(url: str, local_file: str, force_download: bool = False):
    if os.path.exists(local_file) and not force_download:
        print(f"file '{local_file}' already exists, skipping download")
        return
    req = requests.get(url)
    data = bz2.decompress(req.content)
    with open(local_file, 'wb') as fw:
        fw.write(data)


def download_onion_dataset(zenodo_access_token: str, feature_names: Union[str, tuple, list] = ('ivec256',),
                           save_path: str = './', force_download: bool = True,
                           do_not_download_listening_history: bool = False):
    """
    Downloads the onion dataset
    @type zenodo_access_token: Personal Access token. See https://zenodo.org/account/settings/applications/tokens/new/
    @type save_path: path to the folder where to save the raw dataset. Default to "./"
    @type feature_names: name of the feature to download. Default to "ivec256"
    @type force_download: whether to download the file if it already exists
    @type do_not_download_listening_history: whether to download the listening history
                                      (in case it's already available in a preprocessed / filtered form)
    """

    raw_dataset_path = os.path.join(save_path, "raw_dataset")
    features_folder_path = os.path.join(raw_dataset_path, "features")

    os.makedirs(raw_dataset_path, exist_ok=True)
    os.makedirs(features_folder_path, exist_ok=True)

    # Downloading from zenodo
    # https://github.com/lgloege/zenodopy/
    zeno = zenodopy.Client(token=zenodo_access_token, sandbox=False)
    zeno.bucket = ONION_ZENODO_BUCKET

    print('Downloading the onion dataset...')

    download_lh = not do_not_download_listening_history
    if download_lh:
        print('Downloading interaction data...')
        soft_download_zenodo(zeno, 'userid_trackid_timestamp.tsv.bz2', raw_dataset_path, force_download)
        print('Decompressing interaction data...')
        soft_extract_bz2(source_file=os.path.join(raw_dataset_path, "userid_trackid_timestamp.tsv.bz2"),
                         target_file=os.path.join(raw_dataset_path, "userid_trackid_timestamp.tsv"),
                         override=force_download)
    else:
        print('Not downloading interaction data because of parameter...')

    print('Downloading user data...')
    soft_download_bz2(LFM2B_2020_USER_DATASET_LINK, os.path.join(raw_dataset_path, "users.tsv"), force_download)

    print('Downloading the item features...')
    if isinstance(feature_names, str):
        feature_names = [feature_names]

    for feature_name in feature_names:
        print(f'Downloading "{feature_name}"...')
        feature_file = 'id_' + feature_name + '.tsv.bz2'
        soft_download_zenodo(zeno, feature_file, features_folder_path, force_download)

        print(f'Decompressing "{feature_name}"...')
        soft_extract_bz2(source_file=os.path.join(features_folder_path, f"id_{feature_name}.tsv.bz2"),
                         target_file=os.path.join(features_folder_path, f"id_{feature_name}.tsv"),
                         override=force_download)


def download_delivery_hero_sg_dataset(save_path: str = './'):
    """
    Downloads the Delivery Hero 2023 Dataset for Singapore
    https://dl.acm.org/doi/10.1145/3604915.3610242
    @type save_path: path to the folder where to save the raw dataset. Default to "./"
    """

    if not os.path.exists(os.path.join(save_path, 'raw_dataset')):
        os.makedirs(os.path.join(save_path, 'raw_dataset'))

    # Downloading
    print("Downloading the dataset...")
    dataset_zip_name = os.path.join(save_path, 'data_sg.zip')
    gdown.download(id=DELIVERY_HERO_SINGAPORE_DATASET_FILE_ID, output=dataset_zip_name)

    # Unzipping
    with zipfile.ZipFile(dataset_zip_name, 'r') as zipr:
        zipr.extractall(save_path)

    os.remove(dataset_zip_name)
    shutil.rmtree(os.path.join(save_path, '__MACOSX'))

    os.rename('data_sg', 'raw_dataset')

    print('Dataset downloaded')


def download_amazonvid2018_dataset(save_path: str = './'):
    """
    Downloads the Amazon 2018 VideoGame dataset
    @type save_path: path to the folder where to save the raw dataset. Default to "./"
    """

    if not os.path.exists(os.path.join(save_path, 'raw_dataset')):
        os.makedirs(os.path.join(save_path, 'raw_dataset'))

    # Downloading
    print("Downloading the dataset...")

    req = requests.get(AMAZONVID2018_DATASET_LINK, verify=False)
    file_name = os.path.join(save_path, "raw_dataset", "Video_Games.csv")
    with open(file_name, 'wb') as fw:
        fw.write(req.content)


def download_kuai_dataset(save_path: str = './', force_download: bool = False):
    """
    Downloads the KuaiRec dataset
    https://kuairec.com/

    @type save_path: path to the folder where to save the raw dataset. Default to "./"
    @type force_download: whether to download the dataset if it exists already
    """
    dataset_zip_name = os.path.join(save_path, 'dataset.zip')
    if force_download or not os.path.isfile(dataset_zip_name):
        print('downloading dataset')
        gdown.download(id=KUAIREC_DATASET_FILE_ID, output=dataset_zip_name)

        print('extracting dataset')
        with zipfile.ZipFile(dataset_zip_name, 'r') as zipr:
            zipr.extractall(save_path)

        print('dataset download complete')
    else:
        print('dataset already downloaded')

def download_and_decompress_gz(req, gz_file_name, file_name):
    with open(gz_file_name, 'wb') as fw:
        fw.write(req.content)

    with gzip.open(gz_file_name, 'rb') as fr:
        with open(file_name, 'wb') as fw:
            shutil.copyfileobj(fr, fw)
def download_amazon2024_dataset(save_path: str = './', category_official: str = 'All_Beauty', category_short: str = 'beauty', donwload_meta=True, verified_purchase=False):
    """
    Downloads the Amazon 2024 dataset of specified category
    @type save_path: path to the folder where to save the raw dataset. Default to "./"
    @type category: category of the dataset to download. It has to be the same name category
    of the official Amazon2024 naming. Default to "All_Beauty"
    """


    # Downloading
    print("Downloading the dataset...")
    if verified_purchase:
        category_short = f'{category_short}_verified'
        downloaded_file = f'{category_official}.jsonl.gz'
        file_name = os.path.join(save_path, f"raw_dataset/{category_short}", f"{category_official}.jsonl")

    else:
        downloaded_file = f'{category_official}.csv.gz'
        file_name = os.path.join(save_path, f"raw_dataset/{category_short}", f"{category_official}.csv")

    if not os.path.exists(os.path.join(save_path, 'raw_dataset', category_short)):
        os.makedirs(os.path.join(save_path, 'raw_dataset', category_short))

    data_link = f"{AMAZON2024_LINK}/{downloaded_file}"
    req = requests.get(data_link, verify=False)

    gz_file_name = os.path.join(save_path, f"raw_dataset/{category_short}", downloaded_file)
    download_and_decompress_gz(req, gz_file_name, file_name)


    if donwload_meta:
        downloaded_file = f'meta_{category_official}.jsonl.gz'
        data_link = f"{AMAZON2024_META_LINK}/{downloaded_file}"
        req = requests.get(data_link, verify=False)

        file_name = os.path.join(save_path, f"raw_dataset/{category_short}", f"meta_{category_official}.jsonl")
        gz_file_name = os.path.join(save_path, f"raw_dataset/{category_short}", downloaded_file)
        download_and_decompress_gz(req, gz_file_name, file_name)


    print("dataset download complete")
