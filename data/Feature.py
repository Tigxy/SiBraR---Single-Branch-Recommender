import numpy as np
from typing import List
from ast import literal_eval
from collections.abc import Sequence

import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

import torch

from data.config_classes import FeatureDefinition, FeatureType, ProcessingType


def create_padded_array(lists: List[List], padding_idx, width=None):
    """
    Creates a padded array of a list of lists that possible contain a different number of items each.
    """
    max_length = max(map(len, lists))
    if width is None:
        width = max_length
    elif max_length > width:
        raise ValueError(
            f'Specified width is smaller than the maximum of samples in a sublist ({width} < {max_length})')
    return np.array([li + [padding_idx] * (width - len(li)) for li in lists])


class Feature(Sequence):

    def __init__(self, feature_definition: FeatureDefinition,
                 raw_values: sp.sparray | np.ndarray | list[int | float | str], indices: np.ndarray = None,
                 reference_values: sp.sparray | np.ndarray | list[int | float | str] = None):
        """
        Helper class for different user/item features to ease their handling

        :param feature_definition: the definition of the feature
        :param raw_values: the raw feature values for the individual users/items,
                           can either be numeric values (int/float) or string representations of a list of values
        :param indices (optional): the indices that correspond to the specific features, per default, this is
                                   np.arange(0, n_features)
        :param reference_values (optional): reference values for features, this might be important for
                                            categorical / tag features to know which values actually exist
                                            For example, not all features in the train set might appear in the
                                            validation set (or vice versa).
        """
        super().__init__()

        self.feature_definition = feature_definition
        self._n_values = raw_values.shape[0] if hasattr(raw_values, 'shape') else len(raw_values)
        self._raw_values = raw_values
        self._indices = indices if indices is not None else np.arange(0, self._n_values)
        self._indices_map = {idx: i for i, idx in enumerate(self._indices)}

        if self._n_values != len(self._indices):
            raise ValueError(f'Provided indices must match size of supplied values '
                             f'({self._n_values} != {len(self._indices)})')

        # set based on feature type, see below
        self._dim = None
        self._value = None

        # utilities for categorical features
        self._value_map = None
        self._unique_values = None
        self._encoded_values = None
        self._value_indices_groups = None

        # utilities for tag features
        self._value_lists = None

        match feature_definition.type:
            case FeatureType.CATEGORICAL:
                self._process_as_categorical_feature(reference_values)

            case FeatureType.TAG:
                self._process_as_tag_feature(reference_values)

            case FeatureType.SEQUENCE:
                self._process_as_sequential_feature()

            case FeatureType.DISCRETE | FeatureType.CONTINUOUS:
                self._process_as_numeric_feature()

            case FeatureType.VECTOR | FeatureType.MATRIX:
                self._process_as_vector_or_matrix_feature()

            case _:
                raise ValueError(f"FeatureType '{feature_definition.type}' is not supported")

        # just ensure that all important values are set
        assert self._dim is not None
        assert self._values is not None

    @property
    def values(self):
        # leave as property to allow easy modifications
        return self._values

    @property
    def n_values(self) -> int:
        return self._n_values

    @property
    def dim(self):
        return self._dim

    @property
    def unique_values(self):
        self.ensure_is_of_type([FeatureType.CATEGORICAL, FeatureType.TAG], "unique_values")
        return self._unique_values

    @property
    def n_unique_categories(self):
        self.ensure_is_of_type([FeatureType.CATEGORICAL], "n_unique_categories")
        return len(self._unique_values)

    @property
    def value_map(self) -> dict:
        self.ensure_is_of_type([FeatureType.CATEGORICAL, FeatureType.TAG], "value_map")
        return self._value_map

    @property
    def reverse_value_map(self) -> dict:
        self.ensure_is_of_type([FeatureType.CATEGORICAL, FeatureType.TAG], "reverse_value_map")
        return {v: k for k, v in self._value_map.items()}

    def get_labels(self, values):
        self.ensure_is_of_type([FeatureType.CATEGORICAL], "get_labels")
        return np.vectorize(self.reverse_value_map.__getitem__)(values)

    @property
    def value_indices_groups(self):
        self.ensure_is_of_type([FeatureType.CATEGORICAL, FeatureType.TAG], "value_indices_groups")
        return self._value_indices_groups

    @property
    def value_counts(self):
        self.ensure_is_of_type([FeatureType.CATEGORICAL, FeatureType.TAG], "value_counts")
        return {k: len(v) for k, v in self.value_indices_groups.items()}

    def __getitem__(self, i):
        if isinstance(i, np.ndarray):
            # save original shape and flatten indices to allow any shape of indices
            indices_shape = i.shape
            i = i.reshape(-1)

            vec = np.vectorize(self._indices_map.__getitem__)(i)
            values = self.values[vec]

            if isinstance(values, (sp.csr_matrix, sp.csc_matrix, sp.coo_matrix)):
                values = values.toarray()

            # ensure that we do not transform features that are only one dimensional
            if self.dim > 0:
                # reshape to original size
                values = values.reshape(indices_shape + (-1,))

            return values

        elif isinstance(i, torch.Tensor):
            # perform implicit conversion of values to torch.Tensor if this was the type of the index
            values = self.__getitem__(i.numpy(force=True))
            return torch.tensor(values, device=i.device)

        elif isinstance(i, int):
            return self.values[self._indices_map[i]]

        elif isinstance(i, slice):
            raise NotImplementedError('Slicing of features is not yet supported due to difficulties '
                                      'with indices and their mappings to the underlying features.')

        else:
            raise IndexError(f'Indexing is not supported for type {type(i)}.')

    def __len__(self) -> int:
        return self._n_values

    def __repr__(self):
        return str(self)

    def __str__(self):
        str_representation = (f"Feature("
                              f"name={self.feature_definition.name}, "
                              f"type={self.feature_definition.type}, "
                              f"number={self.n_values}, "
                              f"dim={self.dim}"
                              )

        if self.feature_definition.type in (FeatureType.CATEGORICAL, FeatureType.TAG):
            str_representation += f" (counts: {self.value_counts})"

        return str_representation

    def _process_as_numeric_feature(self):
        if isinstance(self._raw_values, list):
            self._values = np.array(self._raw_values)
        else:
            self._values = self._raw_values
        self._values = np.array(self._raw_values)
        self._dim = 1

    def _process_as_categorical_feature(self, reference_values=None):
        # the unique values of the categorical feature
        unique_values = set(self._raw_values)

        # ensure that we cover all categories that we might eventually encounter
        if reference_values is not None:
            unique_ref_values = set(reference_values)
            unique_values = unique_values.union(unique_ref_values)

        # ... sorted for reproducibility
        self._unique_values = sorted(tuple(unique_values))

        # create a mapping for each unique value to an integer label
        self._value_map = {lbl: i for i, lbl in enumerate(self._unique_values)}

        # map raw values to their corresponding labels
        self._values = np.array([self.value_map[lbl] for lbl in self._raw_values], dtype=int)
        self._dim = 0

        # for utility, retrieve which user (item) indices belong to which categorical group
        self._value_indices_groups = {lbl: np.argwhere(self._values == self._value_map[lbl]).flatten()
                                      for lbl in self._unique_values}

        # optionally transform feature by one-hot encoding it
        if self.feature_definition.preprocessing == ProcessingType.ONE_HOT:
            self._dim = len(self._unique_values)
            ohe = OneHotEncoder(categories=self.unique_values)
            self._values = np.array(ohe.fit_transform(self._raw_values))

    def _process_as_tag_feature(self, reference_values=None):
        if self.feature_definition.tag_split_sep is None:
            raise ValueError(f'For tag feature "{self.feature_definition.name}" a separator (tag_split_sep) '
                             f'for the individual has to be provided. For genre tags "action|romance" '
                             f'this would be "|".')

        raw_tags = [set(v.split(self.feature_definition.tag_split_sep)) for v in self._raw_values]

        # the unique values of the tag feature
        unique_values = set().union(*raw_tags)

        # ensure that we cover all tags that we might eventually encounter
        if reference_values is not None:
            raw_ref_tags = [set(v.split(self.feature_definition.tag_split_sep)) for v in reference_values]
            unique_ref_values = set().union(*raw_ref_tags)
            unique_values = unique_values.union(unique_ref_values)

        # ... sorted for reproducibility
        self._unique_values = sorted(tuple(unique_values))

        # create a mapping for each unique value to an integer label
        self._value_map = {lbl: i for i, lbl in enumerate(self._unique_values)}

        # map raw values to their corresponding labels
        self._value_lists = [[self.value_map[tag] for tag in tags] for tags in raw_tags]
        self._values = create_padded_array(self._value_lists, padding_idx=len(self._unique_values))

        # for utility, retrieve which user (item) indices are assigned which tags
        self._value_indices_groups = {tag: [i for i, tgs in enumerate(self._value_lists) if self._value_map[tag] in tgs]
                                      for tag in self._unique_values}

        # dim is number of labels, as this is the maximum tags a feature sample could have
        self._dim = len(self._unique_values)

        # optionally transform feature by multi-hot encoding it
        if self.feature_definition.preprocessing == ProcessingType.MULTI_HOT:
            self._dim = len(self._unique_values)
            mlb = MultiLabelBinarizer(classes=self._unique_values)
            self._values = np.array(mlb.fit_transform(raw_tags))

    def _process_as_sequential_feature(self):
        # evaluate string representations of some feature as a Python datatype
        # note: literal_eval is safe to use as it will throw an error if the data is not of a valid
        #       python datatype (https://stackoverflow.com/q/15197673)
        self._values = np.stack([literal_eval(val) for val in self._raw_values], axis=0)
        self._dim = self._values.shape[1]

    def _process_as_vector_or_matrix_feature(self):
        if isinstance(self._raw_values, list):
            # for list of vectors, transform them into a proper 2d array for more efficient usage
            self._values = np.stack(self._raw_values, axis=0)
        else:
            # otherwise use them as is
            self._values = self._raw_values

        # retrieve proper dimension of feature
        self._dim = self._values.shape[1:]  # for 2d+ features
        if len(self._dim) == 1:
            self._dim = self._dim[0]  # for 1d features

    def ensure_is_of_type(self, feature_types: FeatureType | List[FeatureType], method_name: str):
        if isinstance(feature_types, FeatureType):
            feature_types = [feature_types]

        if self.feature_definition.type not in feature_types:
            raise TypeError(f'Only features of type {feature_types} support "{method_name}"')
