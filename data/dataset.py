import os
import yaml
import logging
import numpy as np
import pandas as pd
from scipy import sparse as sp

import torch
from torch.autograd.profiler import record_function
from torch.utils import data

from data.Feature import Feature
from data.data_preprocessing_utils import load_split_features, load_all_features
from data.preprocessing_config_classes import DataPreprocessingConfig, ColdStartType
from data.sampling import negative_sample_uniform, negative_sample_popular, negative_sample_uniform_recbole
from data.config_classes import (RecDatasetConfig, TrainDatasetConfig, TrainUserRecDatasetConfig,
                                 FeatureDefinition, InteractionDatasetConfig)

"""
The following classes are used to supply the recommender system data to the different methods. In 'data_path', there 
should be the following csv files:
- user_idxs.csv: containing at least the column `user_idx` which is the row index of the user in the interaction matrix.
        Possibly, the file also contains the 'id' used in the original dataset. Used in Train and Eval.
- item_idxs.csv: containing at least the column `item_idx` which is the column index of the item in the interaction matrix.
        Possibly, the file also contains the 'id' used in the original dataset. Used in Train and Eval.
- listening_history_train.csv: containing at least the columns `user_idx` and `item_idx` which corresponds to the entries
        in the interaction matrix used for training. Additional columns are allowed. Used in Train.
- listening_history_val.csv: same as listening_history_train.csv but contains the data used for validation. Used in
        Eval when split_set == val.
- listening_history_test.csv: same as listening_history_train.csv but contains the data used for test. Used in Eval 
        when split_set == test.
"""


class RecDataset(data.Dataset):
    """
    Dataset to hold Recommender System data in the format of a pandas dataframe.
    """

    def __init__(self, config: RecDatasetConfig):
        """
        """
        assert config.split_set in ['train', 'val', 'test'], f'<{config.split_set}> is not a valid value for split set!'
        self._config = config
        self.data_path = config.dataset_path
        self.split_set = config.split_set
        self.model_requires_train_interactions = config.model_requires_train_interactions
        self.keep_listening_histories_in_memory = config.keep_history_in_memory

        self.is_train_split = config.split_set == 'train'
        self.is_eval_split = config.split_set in {'val', 'test'}

        self._preprocessing_config = self._load_preprocessing_config()
        self.cold_start_type = self._preprocessing_config.split.cold_start_type
        self.is_cold_start_user = self.cold_start_type in [ColdStartType.User, ColdStartType.Both]
        self.is_cold_start_item = self.cold_start_type in [ColdStartType.Item, ColdStartType.Both]
        self.is_cold_start_dataset = self.is_cold_start_user or self.is_cold_start_item

        # attributes populated by 'self._load_data()'
        self.n_users = None
        self.n_items = None
        self.n_interactions = None
        self.users_in_split = None
        self.items_in_split = None
        self.listening_history = None
        self.listening_history_train = None

        self.n_user_groups = 0  # optional
        self.user_to_user_group = None  # optional

        self._load_data()

        # definition and loading of user features
        self.user_feature_definitions = config.user_feature_definitions or []
        self.user_feature_names = [f.name for f in self.user_feature_definitions]
        self.user_features = self._load_features(entity="user", feature_definitions=self.user_feature_definitions,
                                                 split=self.split_set)

        # definition and loading of item features
        self.item_feature_definitions = config.item_feature_definitions or []
        self.item_feature_names = [f.name for f in self.item_feature_definitions]
        self.item_features = self._load_features(entity="item", feature_definitions=self.item_feature_definitions,
                                                 split=self.split_set)

        self.features = {
            'user': self.user_features,
            'item': self.item_features
        }
        self.feature_names = {
            'user': self.user_feature_names,
            'item': self.item_feature_names
        }

        logging.info(f'Built {self.name} module \n'
                     f'- data_path: {self.data_path} \n'
                     f'- split_set: {self.split_set} \n'
                     f'- n_users: {self.n_users} \n'
                     f'- n_items: {self.n_items} \n'
                     f'- n_interactions: {self.n_interactions} \n'
                     f'- n_user_groups: {self.n_user_groups} \n'
                     f'- user_features: {self.user_feature_names} \n'
                     f'- item_features: {self.item_feature_names} \n'
                     )

    @property
    def name(self):
        return self.__class__.__name__

    def _load_data(self):
        logging.info('Loading data')

        user_idxs = pd.read_csv(os.path.join(self.data_path, 'user_idxs.csv'))
        item_idxs = pd.read_csv(os.path.join(self.data_path, 'item_idxs.csv'))
        self._process_user_groups(user_idxs)

        self.n_users = len(user_idxs)
        self.n_items = len(item_idxs)

        lhs = self._load_listening_history(self.split_set)
        if self.keep_listening_histories_in_memory:
            # listening history eats up quite a bit of memory, thus don't save it if not required
            self.listening_history = lhs

        # for regular datasets, all their users can be in any of the splits
        # using the total number of users ensures that even if some user is by chance missing in a split,
        # the algorithms working on the data do not crash
        self.users_in_split = np.array(sorted(lhs['user_idx'].unique())) if self.is_cold_start_dataset else user_idxs[
            'user_idx'].to_numpy()
        self.items_in_split = np.array(sorted(lhs['item_idx'].unique())) if self.is_cold_start_dataset else item_idxs[
            'item_idx'].to_numpy()

        self.n_interactions = len(lhs)
        self.n_users_in_split = len(self.users_in_split)
        self.n_items_in_split = len(self.items_in_split)

        # keeping full size matrix for now (even if not all users/items are in the dataset)
        # might want to be changed in a later refactoring
        self.interaction_matrix = self._generate_matrix_from_history(lhs, self.n_users, self.n_items)

        # In the cold start scenario, we still need access to the training interactions as basis
        # for recommendations of known users & items
        if self.model_requires_train_interactions:
            # conditionally set reference to already loaded dataset to not having the same data twice in memory
            train_lhs = lhs if self.is_train_split else self._load_listening_history('train')
            if self.keep_listening_histories_in_memory:
                self.listening_history_train = train_lhs

            self.train_users = np.array(sorted(train_lhs["user_idx"].unique())) if self.is_cold_start_dataset else \
                user_idxs['user_idx'].to_numpy()
            self.train_items = np.array(sorted(train_lhs["item_idx"].unique())) if self.is_cold_start_dataset else \
                item_idxs['item_idx'].to_numpy()

            self.n_train_users = len(self.train_users)
            self.n_train_items = len(self.train_items)
            self.interaction_matrix_train = self._generate_matrix_from_history(train_lhs, self.n_users, self.n_items)

        logging.info('End loading data')

    @staticmethod
    def _generate_matrix_from_history(listening_history: pd.DataFrame, n_users: int, n_items: int,
                                      dtype: type = np.int8, sparse_type: str = 'coo'):
        sparse_cls_map = {
            'coo': sp.coo_matrix,
            'csr': sp.csr_matrix,
            'csc': sp.csc_matrix
        }

        if sparse_type not in sparse_cls_map:
            raise ValueError(f'provided sparse type ("{sparse_type}") not supported. '
                             f'Choose one from {tuple(sparse_cls_map.keys())} instead.')

        data = np.ones(len(listening_history), dtype=dtype)
        row_indices = listening_history["user_idx"]
        column_indices = listening_history["item_idx"]

        return sparse_cls_map[sparse_type]((data, (row_indices, column_indices)),
                                           shape=(n_users, n_items))

    def _process_user_groups(self, user_idxs):
        # Optional grouping of the users
        # TODO: this is deprecated due to the general support of user features
        if 'group_idx' in user_idxs.columns:
            self.user_to_user_group = user_idxs[['user_idx', 'group_idx']].set_index('user_idx').sort_index().group_idx
            self.user_to_user_group = torch.Tensor(self.user_to_user_group)
            self.n_user_groups = user_idxs.group_idx.nunique()

    def _load_preprocessing_config(self) -> DataPreprocessingConfig:
        with open(os.path.join(self.data_path, 'used_config.yaml'), 'r') as fh:
            config = yaml.safe_load(fh)
        return DataPreprocessingConfig.from_dict(config)

    def _load_listening_history(self, split_set: str):
        return pd.read_csv(os.path.join(self.data_path, f'listening_history_{split_set}.csv'))

    def _load_features(self, entity: str, feature_definitions: list[FeatureDefinition], split: str) \
            -> dict[str, Feature]:
        """
        Loads the different features for their corresponding entities (users / items).
        While 'FeatureType.VECTOR' and 'FeatureType.MATRIX' data is stored in separate files,
        all other FeatureTypes are stored in a single file.

        Naming:
            - for feature types 'VECTOR' and 'MATRIX': '{entity_name}_{split}_{feature_name}.npz'
            - else: '{entity_name}_{split}_features.csv'

        :param entity: The entity to load the features for. Either 'user' or 'item'
        :param feature_definitions: The definitions of the features to load. These are the only features that
                                    will be available later on in the dataset.
        """
        logging.info(f"Loading features for '{entity}'")

        all_features = load_all_features(self.data_path, entity, feature_definitions)

        # during training, we also need access to validation features as these splits go hand in hand
        splits = (split, 'val') if self.is_train_split else (split,)
        raw_features = load_all_features(self.data_path, entity, feature_definitions, splits)

        features = {}
        for fd in feature_definitions:
            if fd.name in raw_features.tabular_feature_names:
                # create feature for the specific column in the table
                # provide reference values as the feature might be categorical or a collection of tags, so that
                # we don't miss out on any categories / tags that are not present in the current split
                features[fd.name] = Feature(fd, raw_features.tabular_features[fd.name],
                                            indices=raw_features.tabular_features[f'{entity}_idx'],
                                            reference_values=all_features.tabular_features[fd.name])
            else:
                md_feature = raw_features.multidimensional_features[fd.name]
                features[fd.name] = Feature(fd, md_feature.values, indices=md_feature.indices)

        logging.info(f"Done loading features")
        return features

    def __len__(self):
        raise NotImplementedError("RecDataset does not support __len__ or __getitem__. Please use TrainRecDataset for"
                                  "training or FullEvalDataset for evaluation.")

    def __getitem__(self, index):
        raise NotImplementedError("RecDataset does not support __len__ or __getitem__. Please use TrainRecDataset for"
                                  "training or FullEvalDataset for evaluation.")


class InteractionRecDataset(RecDataset):
    def __init__(self, config: InteractionDatasetConfig):
        super().__init__(config)

        self.model_requires_item_interactions = config.model_requires_item_interactions

        # create sparse matrices in the different formats, as they are faster
        # in different aspects, e.g., row and column slicing
        self.user_sampling_matrix = sp.csr_matrix(self.interaction_matrix)
        if self.model_requires_train_interactions:
            self.user_sampling_matrix_train = sp.csr_matrix(self.interaction_matrix_train)
            if self.model_requires_item_interactions:
                self.item_sampling_matrix_train = sp.csr_matrix(self.interaction_matrix_train.T)

    def get_user_interactions(self, indices: int | np.ndarray | torch.Tensor):
        return self._get_interactions('user', indices)

    def get_user_interaction_vectors(self, indices: int | np.ndarray | torch.Tensor):
        return self._get_interaction_vectors('user', indices)

    def get_item_interactions(self, indices: np.ndarray | torch.Tensor):
        if not self.model_requires_item_interactions:
            raise ValueError(f'get_item_interactions() is not supported for '
                             f'"model_requires_item_interactions=False"')
        return self._get_interactions('item', indices)

    def get_item_interaction_vectors(self, indices: int | np.ndarray | torch.Tensor):
        if not self.model_requires_item_interactions:
            raise ValueError(f'get_item_interaction_vectors() is not supported for '
                             f'"model_requires_item_interactions=False"')
        return self._get_interaction_vectors('item', indices)

    def get_user_feature(self, feature_name: str, indices: np.ndarray | torch.Tensor):
        return self._get_feature('user', feature_name, indices)

    def get_item_feature(self, feature_name: str, indices: np.ndarray | torch.Tensor):
        return self._get_feature('item', feature_name, indices)

    def get_features(self, entity: str, feature_names: list[str], indices: np.ndarray | torch.Tensor):
        return {f: self._get_feature(entity, f, indices) for f in feature_names}

    def get_item_features(self, feature_names: list[str], indices: np.ndarray | torch.Tensor):
        return self.get_features('item', feature_names, indices)

    def get_user_features(self, feature_names: list[str], indices: np.ndarray | torch.Tensor):
        return self.get_features('user', feature_names, indices)

    @staticmethod
    def _get_numpy_array(a: torch.Tensor):
        if isinstance(a, torch.Tensor):
            return a.cpu().numpy(), a.device
        return a, None

    def _get_interactions(self, entity: str, indices: int | np.ndarray | torch.Tensor):
        matrix = self.user_sampling_matrix_train if entity == 'user' else self.item_sampling_matrix_train

        # reshaping 'indices' would not make a lot of sense, as each user/item might have a different number
        # of interactions, therefore no new array/tensor could anyway be formed
        indices, d = self._get_numpy_array(indices)
        interaction_indices = matrix[indices].indices

        return interaction_indices

    def _get_interaction_vectors(self, entity: str, indices: int | np.ndarray | torch.Tensor):
        matrix = self.user_sampling_matrix_train if entity == 'user' else self.item_sampling_matrix_train
        indices, d = self._get_numpy_array(indices)

        # save original shape and flatten indices to allow any shape of indices
        indices_shape = indices.shape
        indices = indices.reshape(-1)

        interactions = matrix[indices].toarray().astype(float)
        interactions = interactions.reshape(indices_shape + (-1,))

        if d is not None:
            return torch.tensor(interactions, device=d)
        return interactions

    def _get_feature(self, entity: str, feature_name: str, indices: np.ndarray | torch.Tensor):
        return self.features[entity][feature_name][indices]


class TrainRecDataset(InteractionRecDataset):
    """
    Dataset to hold Recommender System data and train collaborative filtering algorithms. It allows iteration over the
    dataset of positive interaction. It also stores the item popularity distribution over the training data.

    Additional notes:
    The data is loaded twice. Once the data is stored in a COO matrix to easily iterate over the dataset. Once in a CSR
    matrix to carry out fast negative sampling with the user-wise slicing functionalities (see also collate_fn in data/dataloader.py)
    """

    def __init__(self, config: TrainDatasetConfig):
        """
        """
        super().__init__(config)

        self.n_negative_samples = config.n_negative_samples
        self.negative_sampling_strategy = config.negative_sampling_strategy
        self.use_dataset_negative_sampler = config.use_dataset_negative_sampler
        self.sampling_popularity_squashing_factor = config.sampling_popularity_squashing_factor
        self.model_requires_pop_distribution = config.model_requires_pop_distribution

        # negative sampling computations can be accelerated by retrieving item indices beforehand
        # (1000x over csr matrix and over 2x faster than a lil matrix, which implies that this extra memory
        # cost is worth it)
        self.sampling_row_indices = [self.user_sampling_matrix[i].indices for i in range(self.n_users)]

        if self.negative_sampling_strategy == 'popular' or self.model_requires_pop_distribution:
            # to save memory, only calculate pop distribution if it's actually necessary
            self.pop_distribution = self._get_pop_distribution()

        logging.info(f'Built {self.name} module \n')

    def _get_pop_distribution(self):
        item_popularity = np.array(self.user_sampling_matrix.sum(axis=0)).flatten()
        return item_popularity / item_popularity.sum()

    def _get_negative_samples(self, user_index: int):
        match self.negative_sampling_strategy:
            case 'uniform':
                return negative_sample_uniform(choices=self.items_in_split, size=self.n_negative_samples,
                                               positive_indices=self.sampling_row_indices[user_index])
            case 'uniform_recbole':
                return negative_sample_uniform_recbole(choices=self.items_in_split, size=self.n_negative_samples,
                                                       positive_indices=self.sampling_row_indices[user_index])
            case 'popular':
                return negative_sample_popular(choices=self.items_in_split, size=self.n_negative_samples,
                                               popularity_distribution=self.pop_distribution,
                                               squashing_factor=self.sampling_popularity_squashing_factor,
                                               positive_indices=self.sampling_row_indices[user_index])
            case _:
                raise ValueError(f'Sampling strategy "{self.negative_sampling_strategy}" not yet supported.')

    def __len__(self):
        return self.interaction_matrix.nnz

    def __getitem__(self, index):
        user_idx = self.interaction_matrix.row[index].astype('int64')
        item_idx = self.interaction_matrix.col[index].astype('int64')
        item_label = 1.

        if self.use_dataset_negative_sampler:
            with record_function("sampling"):
                item_label = np.array([item_label])

                negative_item_indices = self._get_negative_samples(user_idx)
                negative_item_labels = np.zeros_like(negative_item_indices, dtype=float)

                item_indices = np.concatenate([[item_idx], negative_item_indices])
                item_labels = np.concatenate([item_label, negative_item_labels])
                return user_idx, item_indices, item_labels

        return user_idx, item_idx, item_label


class FullEvalDataset(InteractionRecDataset):
    """
    Dataset to hold Recommender System data and evaluate collaborative filtering algorithms. It allows iteration over
    all the users and compute the scores for all items (FullEvaluation). It also holds data from training and validation
    that needs to be excluded from the evaluation:
    During validation, items in the training data for a user are excluded as labels
    During test, items in the training data and validation for a user are excluded as labels
    """

    def __init__(self, config: InteractionDatasetConfig):
        # we need access to train interactions, thus set flag to True
        config.model_requires_train_interactions = True
        super().__init__(config)
        self.exclude_data = self._get_interacted_mask()

        logging.info(f'Built {self.name} module')

    def _get_interacted_mask(self):
        assert self.user_sampling_matrix_train is not None

        # create empty sparse matrix (there doesn't seem to be an easier way to do this)
        mask = sp.csr_matrix(([], [[], []]),
                             shape=self.user_sampling_matrix_train.shape,
                             dtype=self.user_sampling_matrix_train.dtype)

        # for evaluation, ignore the interactions that are in the training set
        if self.split_set != 'train':
            mask += self.user_sampling_matrix_train

        # for testing, also ignore interactions from the validation set
        if self.split_set == 'test':
            val_lhs = self._load_listening_history('val')
            mask += self._generate_matrix_from_history(val_lhs, self.n_users, self.n_items,
                                                       dtype=bool, sparse_type='csr')

        # we actually only care about the items in our split, all others are irrelevant for evaluation
        # we currently don't filter the users that are not in our split, as this might cause confusion
        #   with the rest of framework (self.interaction_matrix still contains all users and items...)
        #   think about this in the future  # TODO check how to handle
        return mask[:, self.items_in_split].astype(bool)

    def __len__(self):
        # as we consider iterating over the users, we only want to do so for those that are actually in our split
        # this is important for the different cold-start scenarios
        return self.n_users_in_split

    def __getitem__(self, idx):
        # index does not correspond to actual user index, thus retrieve that one first
        user_idx = self.users_in_split[idx]

        # get the true interactions of the user
        labels = self.user_sampling_matrix[user_idx][:, self.items_in_split].toarray().squeeze().astype('float32')

        # return everything
        return user_idx, self.items_in_split, labels


class ECFTrainRecDataset(TrainRecDataset):

    def __init__(self, config: TrainDatasetConfig):
        """
        """

        super().__init__(config)

        self.tag_matrix = None
        self._prepare_tag_data()

        logging.info(f'Built {self.name} module \n')

    def _prepare_tag_data(self):
        tag_idxs = pd.read_csv(os.path.join(self.data_path, 'tag_idxs.csv'))
        item_tag_idxs = pd.read_csv(os.path.join(self.data_path, 'item_tag_idxs.csv'))

        # Creating tag matrix
        self.tag_matrix = sp.csr_matrix(
            (np.ones(len(item_tag_idxs), dtype=np.int16), (item_tag_idxs.item_idx, item_tag_idxs.tag_idx)),
            shape=(self.n_items, len(tag_idxs))
        )

        tag_frequency = np.array(self.tag_matrix.sum(axis=0)).flatten()

        tag_weight = np.log(self.n_items / (tag_frequency + 1e-6))
        # Applying weight to tags
        self.tag_matrix = self.tag_matrix @ sp.diags(tag_weight)


class TrainUserRecDataset(TrainRecDataset):
    """
    Dataset that iterates over the users. It is used during training in pair with the positive/negative sampler.
    """

    def __init__(self, config: TrainUserRecDatasetConfig):
        super().__init__(config)

        self.n_pos = config.n_pos

        del self.interaction_matrix

        logging.info(f'Built {self.name} module \n'
                     f'- n_pos: {self.n_pos} \n')

    def __len__(self):
        return self.n_users

    def __getitem__(self, user_idx):
        user_data = self.user_sampling_matrix[user_idx].indices
        item_pos_idxs = np.random.choice(user_data, size=self.n_pos, replace=len(user_data) < self.n_pos)
        return user_idx, item_pos_idxs
