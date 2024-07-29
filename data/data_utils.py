import logging
import os.path
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
from dask import dataframe as dd

from torch.utils.data import DataLoader

import algorithms.algorithms_utils
from data.config_classes import TrainDatasetConfig, ExperimentConfig, InteractionDatasetConfig
from data.dataloader import TrainDataLoader, NegativeSampler, NegativeSamplingDataLoader
from data.dataset import TrainRecDataset, FullEvalDataset, ECFTrainRecDataset, RecDataset


def get_dataset_and_loader(conf: ExperimentConfig, split_set: str,
                           retrieve_eval_loader: bool = False) -> (RecDataset, DataLoader):
    """
    Returns the dataloader associated to the configuration in conf
    """
    dataloader_config = conf.train_loader if (split_set == 'train' and not retrieve_eval_loader) else conf.val_loader

    if split_set == 'train' and not retrieve_eval_loader:
        dataset_config = TrainDatasetConfig.from_dict_ext(conf.dataset, split_set=split_set)
        if conf.algorithm_type == algorithms.algorithms_utils.AlgorithmsEnum.ecf:
            dataset_cls = ECFTrainRecDataset
        else:
            dataset_cls = TrainRecDataset
        dataset = dataset_cls(dataset_config)

        if dataset_config.use_dataset_negative_sampler:
            dataloader = DataLoader(dataset, **dataloader_config.as_dict())
        else:
            match dataset_config.negative_sampling_strategy:
                case 'uniform':
                    sampler = NegativeSampler(
                        train_dataset=dataset,
                        n_neg=conf.dataset.get('n_negative_samples', 10),
                        neg_sampling_strategy=conf.dataset.get('negative_sampling_strategy', 'uniform')
                    )
                    dataloader = TrainDataLoader(sampler, dataset, **dataloader_config.as_dict())
                case 'uniform_recbole':
                    dataloader = NegativeSamplingDataLoader(dataset, **dataloader_config.as_dict())
                case _:
                    raise ValueError(f'The sampling strategy "{dataset_config.negative_sampling_strategy}" '
                                     f'is not supported')

    elif split_set in {'train', 'val', 'test'}:
        dataset_config = InteractionDatasetConfig.from_dict_ext(conf.dataset, split_set=split_set)
        dataset_cls = FullEvalDataset
        dataset = dataset_cls(dataset_config)
        dataloader = DataLoader(dataset, **dataloader_config.as_dict())
    else:
        raise ValueError(f"split_set value '{split_set}' is invalid! Please choose from [train, val, test]")

    logging.info(f"Built {split_set} DataLoader module: {dataloader_config}")
    return dataset, dataloader


def preprocess_feature_data(feature_file: str, entity_name: str,
                            entity_indices: List[int] | np.ndarray,
                            indices_lookup: pd.DataFrame):
    feat = dd.read_csv(feature_file, sep='\t')
    if 'ID' in feat.columns:
        feat = feat.drop(columns=['id'])
        column_renames = {'ID': entity_name}
        feat = feat.rename(columns=column_renames)
    else:
        column_renames = {'id': entity_name}
        feat = feat.rename(columns=column_renames)

    feat = feat[feat[entity_name].isin(entity_indices)].compute()
    feat = indices_lookup.merge(feat, how='inner')
    feat = feat.drop(columns=[entity_name])
    feat = feat.set_index(f'{entity_name}_idx').sort_index(ascending=True)  # sort based on index for lookup
    return feat.to_numpy()


def split_feature_id_from_data(df: pd.DataFrame, id_column: str, columns_to_ignore: List[str] = None):
    columns_to_drop = [id_column] + (columns_to_ignore or [])
    return df[id_column].to_numpy(), df.drop(columns=columns_to_drop).to_numpy()


def store_feature_data(feature_name: str, feature_data: pd.DataFrame, entity_name: str, split: str,
                       storage_dir: str, store_as_numpy: bool = False):
    os.makedirs(storage_dir, exist_ok=True)

    feature_file_path = os.path.join(storage_dir, f'{entity_name}_{split}_{feature_name}')
    if store_as_numpy:
        feature_file_path += '.npy'
        np.save(feature_file_path, feature_data.to_numpy())
    else:
        feature_file_path += '.tsv'
        feature_data.to_csv(feature_file_path, index=False, sep='\t')


def merge_dicts(first: dict, second: dict):
    """
    Merges two dictionaries and all their subsequent dictionaries.
    In case both dictionaries contain the same key, which is not another dictionary, the latter one is used.

    This merges in contrast to dict.update() all subdicts and its items
    instead of overriding the former with the latter.
    """
    fk = set(first.keys())
    sk = set(second.keys())
    common_keys = fk.intersection(sk)

    z = {}
    for k in common_keys:
        if isinstance(first[k], dict) and isinstance(second[k], dict):
            z[k] = merge_dicts(first[k], second[k])
        else:
            z[k] = deepcopy(second[k])

    for k in fk - common_keys:
        z[k] = deepcopy(first[k])

    for k in sk - common_keys:
        z[k] = deepcopy(second[k])

    return z
