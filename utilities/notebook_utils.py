import os
import sys
import warnings
import dill as pkl
import numpy as np
import pandas as pd
from itertools import combinations

from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

from tqdm.autonotebook import tqdm

tqdm.pandas()

from data_paths import get_processed_dataset_path
from data.config_classes import DatasetsEnum


def get_country_code_map():
    df = pd.read_csv('../notebooks/country_codes.csv', index_col=False)
    return {alpha_2: name for name, alpha_2 in zip(df['name'], df['alpha-2'])}


def get_onion_users():
    data_path = get_processed_dataset_path(DatasetsEnum.onion)
    # ignore occupation and zip code, as they seem unusable 
    df_users = pd.read_csv(os.path.join(data_path, 'user_features.csv'), index_col=False)
    df_users['gender'] = df_users['gender'].replace({'m': 'male', 'f': 'female', 'n': pd.NA})
    df_users['age'] = df_users['age'].replace({-1: pd.NA})
    df_users['country'] = df_users['country'].replace(get_country_code_map())
    return df_users


def store_embeddings(dataset: DatasetsEnum, entity: str, feature_name: str, indices: np.ndarray, values: np.ndarray):
    data_path = get_processed_dataset_path(dataset)
    feature_file_path = os.path.join(data_path, f'{entity}_{feature_name}.npz')
    np.savez(feature_file_path, indices=indices, values=values)


def store_onion_embeddings(entity: str, feature_name: str, indices: np.ndarray, values: np.ndarray):
    return store_embeddings(DatasetsEnum.onion, entity, feature_name, indices, values)


def batch_samples(samples, batch_size, use_tqdm=True, desc=None):
    """
    Batch a list of samples into smaller batches and yield each batch.

    Parameters:
    - samples (list): The list of samples to be batched.
    - batch_size (int): The desired batch size.
    - use_tqdm (bool): Flag to enable/disable tqdm for progress tracking.

    Yields:
    - Each batch as a list of samples.
    """
    num_samples = len(samples)
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division to ensure all samples are included

    if use_tqdm:
        for i in tqdm(range(num_batches), desc=desc):
            yield samples[i * batch_size:(i + 1) * batch_size]
    else:
        for i in range(num_batches):
            yield samples[i * batch_size:(i + 1) * batch_size]


def get_combinations_and_abbreviations(names: list[str]):
    combs = {}
    for i in range(0, len(names) + 1):
        for comb in combinations(names, i):
            combs[''.join(sorted([c[0] for c in comb]))] = comb
    return combs


def generate_sentences(data: pd.DataFrame, columns: list[str], append_unknown=True):
    sentences = []

    if len(columns) == 0:
        return ['I am human.'] * len(data)

    for index, row in data.iterrows():
        known_infos = []
        unknown_infos = []

        if 'age' in columns:
            if row['age'] is not pd.NA:
                known_infos.append(f"{row['age']} years old")
            else:
                unknown_infos.append('age')

        if 'gender' in columns:
            if row['gender'] is not pd.NA and row['gender'] not in ['male', 'female']:
                known_infos.append(f"{row['gender']}")
            else:
                unknown_infos.append('gender')

        if 'country' in columns:
            if row['country'] is not pd.NA:
                known_infos.append(f"from {row['country']}")
            else:
                unknown_infos.append('country')

        sentence = ''
        if len(known_infos):
            sentence += f'I am {" ".join(known_infos)}.'
        if append_unknown and len(unknown_infos) > 0:
            sentence += f' My {" and ".join(unknown_infos)} {"are" if len(unknown_infos) > 1 else "is"} a secret.'

        if len(sentence) == 0:
            sentence = 'I am human.'

        sentences.append(sentence)

    return sentences


def _store_data(dataset: DatasetsEnum, entity: str, feature_name: str, data: any):
    data_path = get_processed_dataset_path(dataset)
    feature_file_path = os.path.join(data_path, f'{entity}_{feature_name}.pkl')

    with open(feature_file_path, 'wb') as fh:
        pkl.dump(data, fh)


def store_onion_data(entity: str, feature_name: str, data: any):
    return _store_data(DatasetsEnum.onion, entity, feature_name, data)


def split_sequence_indices(num_items, n):
    # Calculate the number of items in each part
    items_per_part = num_items // n
    remainder = num_items % n

    # Initialize variables
    start = 0
    indices = [0]

    # Iterate through each part
    for i in range(n):
        # Calculate the end index for the current part
        end = start + items_per_part + (1 if i < remainder else 0)

        # Add the current end index to the list
        indices.append(end - 1)

        # Update the start index for the next part
        start = end

    return indices


class SilenceStd:
    def __init__(self, std_type='out', silence=True):
        self._std_type = std_type
        self._silence = silence
        self._std = None

    def __enter__(self):
        if self._std_type == 'out':
            self._std = sys.stdout
            sys.stdout = self
        else:
            self._std = sys.stderr
            sys.stderr = self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._std_type == 'out':
            sys.stdout = self._std
        else:
            sys.stderr = self._std

    def write(self, data):
        if not self._silence:
            self._std.write(data)

    def flush(self):
        if not self._silence:
            self._std.flush()


class SilenceStds:
    def __init__(self, silence_stdout=True, silence_stderr=True):
        self.silence_stdout = silence_stdout
        self.silence_stderr = silence_stderr

        self._stdout_silencer = SilenceStd('out', silence=silence_stdout)
        self._stderr_silencer = SilenceStd('err', silence=silence_stderr)

    def __enter__(self):
        self._stdout_silencer.__enter__()
        self._stderr_silencer.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout_silencer.__exit__(exc_type, exc_value, traceback)
        self._stderr_silencer.__exit__(exc_type, exc_value, traceback)
