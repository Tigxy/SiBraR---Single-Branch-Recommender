from typing import List

import numpy as np
import pandas as pd

from data.preprocessing_config_classes import FeatureConfig
from data.preprocessing_data_classes import EntityFeatures, MultiDFeature


def describe_listening_history(listening_history):
    return (f"{len(listening_history):10d} entries, "
            f"{listening_history.user.nunique():7d} users, "
            f"{listening_history.item.nunique():7d} items")


def print_description_listening_history(listening_history: pd.DataFrame, preface: str = "", epilogue: str = "",
                                        verbose: bool = True):
    if verbose:
        message = ""
        if preface:
            message += preface + " "
        message += describe_listening_history(listening_history)
        if epilogue:
            message += " " + epilogue
        print(message)


def print_verbose(*args, verbose: bool = True, **kwargs):
    if verbose:
        print(*args, **kwargs)


def filter_missing(df: pd.DataFrame, column: str):
    """
    Drop samples whose features are missing
    """
    return df.dropna(subset=[column])


def filter_values(df: pd.DataFrame, column: str, values: list):
    """
    Keep only samples whose features are included in some list
    """
    return df[df[column].isin(values)]


def filter_range(df: pd.DataFrame, column: str, min_value: float, max_value: float):
    """
    Keep only values that are located in some range
    """
    return df[df[column].between(min_value, max_value)]


def filter_top_categories(df: pd.DataFrame, column: str, top_n: int):
    """
    Keep only the 'top_n' most occurring categories
    """
    top_values = df[column].value_counts().nlargest(top_n).index
    return filter_values(df, column, top_values)


def filter_tabular_features(entity: str, features_df: pd.DataFrame, features_config: List[FeatureConfig],
                            verbose: bool = True):
    """
    Perform preprocessing of features with their respective configurations
    """
    for feature in features_config:
        for step in feature.preprocessing:
            n_samples_before = len(features_df)
            kind = step.kind

            match kind:
                case 'filter_values':
                    features_df = filter_values(features_df, feature.name, **step.parameters)

                case 'filter_range':
                    features_df = filter_range(features_df, feature.name, **step.parameters)

                case 'filter_missing':
                    features_df = filter_missing(features_df, feature.name)

                case 'filter_top':
                    features_df = filter_top_categories(features_df, feature.name, **step.parameters)

                case _:
                    raise ValueError(f'Preprocessing kind "{kind}" is not supported. '
                                     f'Choose from {["filter_values", "filter_range", "filter_missing", "filter_top"]}')
            n_samples_after = len(features_df)

            print_verbose(f'performed {entity} preprocessing "{kind}" '
                          f'for feature "{feature.name}" '
                          f'with parameters {step.parameters} and dropped '
                          f'{n_samples_before - n_samples_after} {entity}s - {n_samples_after} remain.',
                          verbose=verbose)

    return features_df


def filter_interactions(listening_history: pd.DataFrame, min_interactions: int = 2, verbose: bool = True):
    lhs_count = listening_history.value_counts(subset=['user', 'item'])
    lhs_count = lhs_count[lhs_count >= min_interactions]
    listening_history = listening_history.set_index(['user', 'item']).loc[lhs_count.index]
    listening_history = listening_history.reset_index()
    print_verbose(f'dataset with interactions that happened at least {min_interactions} times:',
                  describe_listening_history(listening_history), verbose=verbose)
    return listening_history


def filter_unique(listening_history: pd.DataFrame, verbose: bool = True):
    listening_history = listening_history.sort_values('timestamp')
    listening_history = listening_history.drop_duplicates(subset=['user', 'item'])
    print_verbose(f'dataset with no duplicate entries:',
                  describe_listening_history(listening_history), verbose=verbose)
    return listening_history


def filter_k_core(lhs: pd.DataFrame, k: int, verbose: bool = True) -> pd.DataFrame:
    """
    Performs core-filtering on the dataset.
    @param lhs: Pandas Dataframe containing the listening records. Has columns ["user", "item"]
    @param k: Threshold for performing the k-core filtering.
    @param verbose: Whether to inform about progress.
    @return: Filtered Dataframe
    """
    print_verbose(f'performing {k}-core filtering', verbose=verbose)
    while True:
        start_number = len(lhs)

        # Item pass
        item_counts = lhs.item.value_counts()
        item_above = set(item_counts[item_counts >= k].index)
        lhs = lhs[lhs.item.isin(item_above)]

        print_verbose(f'-- records after item pass: {len(item_above)} items, {len(lhs)} interactions', verbose=verbose)

        # User pass
        user_counts = lhs.user.value_counts()
        user_above = set(user_counts[user_counts >= k].index)
        lhs = lhs[lhs.user.isin(user_above)]

        print_verbose(f'-- records after user pass: {len(user_above)} items, {len(lhs)} interactions', verbose=verbose)

        if len(lhs) == start_number:
            break

    print_description_listening_history(lhs, preface=f'{k}-core filtering complete. Dataset contains',
                                        verbose=verbose)
    return lhs


def filter_history(listening_history: pd.DataFrame, entity: str, entity_features: EntityFeatures, verbose: bool = True):
    common_indices = get_common_feature_indices(entity, entity_features)
    if common_indices is None:
        print_verbose(f'not filtering listening history based on availability of {entity} features, as no '
                      f'{entity} features exist', verbose=verbose)
        return listening_history

    print_verbose(f'filtering listening history based on availability of {entity} features', verbose=verbose)
    listening_history = listening_history[listening_history[entity].isin(common_indices)]
    print_description_listening_history(listening_history, preface='-- remaining history:', verbose=verbose)
    return listening_history


def filter_based_on_history(listening_history: pd.DataFrame, entity: str, entity_features: EntityFeatures,
                            verbose: bool = True):
    # sorting indices so that we will never mix up to which numpy arrays which index belongs to
    unique_indices = sorted(listening_history[entity].unique())

    print_verbose(f'filter {entity}s based on presence in listening history '
                  f'({len(unique_indices)} unique entries exist)')

    return filter_based_on_indices(entity, entity_features, unique_indices, verbose=verbose)


def filter_based_on_indices(entity: str, entity_features: EntityFeatures, indices: list | set, verbose: bool = True):
    indices = set(indices)

    print_verbose(f'filtering {entity} features based on {len(indices)} indices')

    tabular_features = None
    if entity_features.tabular_features is not None:
        tabular_features = filter_values(entity_features.tabular_features, column=entity, values=list(indices))
    multidimensional_features = filter_multi_d_features(entity_features.multidimensional_features,
                                                        indices=indices, verbose=verbose)
    return EntityFeatures(entity_features.tabular_feature_names, tabular_features, multidimensional_features)


def filter_multi_d_features(multi_d_features: dict[str, MultiDFeature], indices: np.ndarray | pd.Series,
                            verbose: bool = True):
    filtered_features = dict()
    for feature_name, feature in multi_d_features.items():
        # check which features are available for the different indices
        indices = set(indices)
        mask = np.array([(i in indices) for i in feature.indices], dtype=bool)

        print_verbose(f'dropping {sum(~mask)} {feature_name} features, {sum(mask)} remain', verbose=verbose)

        # finally store this to return them
        filtered_features[feature_name] = MultiDFeature(
            # and select only those features that match the mask
            indices=feature.indices[mask],
            values=feature.values[mask]
        )

    return filtered_features


def filter_entities_without_all_features(entity: str, entity_features: EntityFeatures,
                                         verbose: bool = True) -> EntityFeatures:
    # check which samples have all features
    common_ids = get_common_feature_indices(entity, entity_features)
    if common_ids is None:
        print_verbose(f'no {entity} features are available', verbose=verbose)
        return entity_features

    print_verbose(f'keeping only those {len(common_ids)} {entity}s for which all features are available',
                  verbose=verbose)

    # and keep only them
    return filter_based_on_indices(entity, entity_features, common_ids)


def get_common_feature_indices(entity: str, features: EntityFeatures) -> set | None:
    tabular_ids = set(features.tabular_features[entity]) if features.tabular_features is not None else None
    multi_d_ids = get_common_multi_d_indices(features.multidimensional_features)

    if tabular_ids is not None and multi_d_ids is not None:
        return set.intersection(tabular_ids, multi_d_ids)
    elif tabular_ids is not None:
        return set(tabular_ids)
    elif multi_d_ids is not None:
        return set(multi_d_ids)
    else:
        return None


def get_common_multi_d_indices(multi_d_features: dict[str, MultiDFeature]):
    collected_indices = [set(f.indices) for _, f in multi_d_features.items()]
    return set.intersection(*collected_indices) if len(collected_indices) > 0 else None


def update_indices(entity: str, entity_features: EntityFeatures, indices_map: dict | pd.DataFrame,
                   verbose: bool = True):
    print_verbose(f'updating indices for {entity}', verbose=verbose)

    # update indices for tabular features
    if entity_features.tabular_features is not None:
        indices_map_df = indices_map
        if not isinstance(indices_map, pd.DataFrame):
            indices_map_df = pd.DataFrame.from_dict({entity: indices_map.keys(), f'{entity}_idx': indices_map.values()})
        entity_features.tabular_features = indices_map_df.merge(entity_features.tabular_features)

    # update indices for multidimensional features
    indices_map_dict = indices_map
    if not isinstance(indices_map, dict):
        # create dict mapping as it does not exist
        indices_map_dict = {old: new for old, new in zip(indices_map[entity], indices_map[f'{entity}_idx'])}

    for k, v in entity_features.multidimensional_features.items():
        # map the individual indices to their new index
        entity_features.multidimensional_features[k].indices = np.vectorize(indices_map_dict.get)(v.indices)

    return entity_features


def sort_based_on_indices(entity: str, entity_features: EntityFeatures, verbose: bool = True) -> None:
    print_verbose(f'sorting features for {entity}', verbose=verbose)

    if entity_features.tabular_features is not None:
        entity_features.tabular_features.sort_values(by=f'{entity}_idx', inplace=True)

    for k, v in entity_features.multidimensional_features.items():
        sorted_order = np.argsort(v.indices)
        v.indices = v.indices[sorted_order]
        v.values = v.values[sorted_order]
