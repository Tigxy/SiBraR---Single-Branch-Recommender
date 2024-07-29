from typing import List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

from data.data_preprocessing_utils import merge_features
from data.filtering import print_verbose
from data.config_classes import FeatureType
from data.preprocessing_data_classes import AllSplitsData, EntityFeatures
from data.preprocessing_config_classes import FeatureConfig, DataPreprocessingConfig, SplitType


class IdentityTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        return data

    def fit_transform(self, data, y=None, **kwargs):
        return self.fit(data, y).transform(data)


def _normalize_features(entity: str, tr_features: EntityFeatures, vd_features: EntityFeatures,
                        te_features: EntityFeatures, entity_feature_configs: List[FeatureConfig],
                        split_type: SplitType, verbose: bool = True):
    normalization_basis = tr_features
    if split_type in [SplitType.Random, SplitType.Temporal]:
        # for some ways of splitting, it is ok to normalize based on all the features in the dataset
        # because we assume that, except for the interactions, everything about users & items
        # is already known. Moreover, the same users and items will be used in the different splits
        # anyway.
        normalization_basis = merge_features(entity, [tr_features, vd_features, te_features])

    for feature_config in entity_feature_configs:
        feature_name = feature_config.name
        feature_type = feature_config.type

        print_verbose(f'normalizing {entity} feature "{feature_name}"',
                      verbose=verbose and len(feature_config.normalization) > 0)

        for step in feature_config.normalization:
            normalizer = get_normalizer(step.kind, step.parameters)

            match feature_type:
                case FeatureType.CATEGORICAL:
                    raise ValueError(f'Categorical feature "{feature_name}" cannot be normalized')
                case FeatureType.TAG:
                    raise ValueError(f'Tag feature "{feature_name}" cannot be normalized')
                case FeatureType.DISCRETE | FeatureType.CONTINUOUS:
                    # reshape to match specification of sklearn normalizers
                    tr_feature = tr_features.tabular_features[feature_name].to_numpy().reshape(-1, 1)
                    vd_feature = vd_features.tabular_features[feature_name].to_numpy().reshape(-1, 1)
                    te_feature = te_features.tabular_features[feature_name].to_numpy().reshape(-1, 1)

                    # fit normalizer
                    fit_data = normalization_basis.tabular_features[feature_name].to_numpy().reshape(-1, 1)
                    normalizer.fit(fit_data)

                    # normalize and undo reshaping
                    tr_feature = normalizer.transform(tr_feature).reshape(-1)
                    vd_feature = normalizer.transform(vd_feature).reshape(-1)
                    te_feature = normalizer.transform(te_feature).reshape(-1)

                    # save results
                    tr_features.tabular_features = tr_features.tabular_features.assign(**{feature_name: tr_feature})
                    vd_features.tabular_features = vd_features.tabular_features.assign(**{feature_name: vd_feature})
                    te_features.tabular_features = te_features.tabular_features.assign(**{feature_name: te_feature})

                case FeatureType.VECTOR | FeatureType.MATRIX:

                    # fit normalizer
                    fit_data = normalization_basis.multidimensional_features[feature_name].values
                    normalizer.fit(fit_data)

                    # and transform all three split datasets with it
                    tr_features.multidimensional_features[feature_name].values = normalizer.transform(
                        tr_features.multidimensional_features[feature_name].values)
                    vd_features.multidimensional_features[feature_name].values = normalizer.transform(
                        vd_features.multidimensional_features[feature_name].values)
                    te_features.multidimensional_features[feature_name].values = normalizer.transform(
                        te_features.multidimensional_features[feature_name].values)

                case _:
                    raise ValueError(f'Feature "{feature_name}" of type "{feature_type}" cannot be normalized')

    return tr_features, vd_features, te_features


def normalize_features(data: AllSplitsData, features_config: DataPreprocessingConfig, verbose: bool = True):
    """
    Perform normalization of features with their respective configurations
    """
    result = _normalize_features('user', data.tr_data.user_features, data.vd_data.user_features,
                                 data.te_data.user_features, features_config.user_features,
                                 split_type=features_config.split.split_type, verbose=verbose)
    # unpack result into original variables
    data.tr_data.user_features, data.vd_data.user_features, data.te_data.user_features = result

    result = _normalize_features('item', data.tr_data.item_features, data.vd_data.item_features,
                                 data.te_data.item_features, features_config.item_features,
                                 split_type=features_config.split.split_type, verbose=verbose)
    # unpack result into original variables
    data.tr_data.item_features, data.vd_data.item_features, data.te_data.item_features = result

    return data


def get_normalizer(kind: str, parameters: dict):
    # see https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    # for different normalizers
    match kind:
        case "standard":
            # to zero mean and unit variance
            return StandardScaler(**parameters)
        case "minmax":
            # scales into given range, default 0..1
            return MinMaxScaler(**parameters)
        case "robust":
            # similar to StandardScaler, but uses median and interquartile range
            # to be robust against outliers
            return RobustScaler(**parameters)
        case 'normal':
            return Normalizer(**parameters)
        case None:
            # does basically nothing
            return IdentityTransform()
        case _:
            raise ValueError(f'Normalizer kind "{kind}" is not supported. '
                             f'Choose from {["standard", "minmax", "robust", "normal"]} '
                             f'or set it to "None" to disable it')
