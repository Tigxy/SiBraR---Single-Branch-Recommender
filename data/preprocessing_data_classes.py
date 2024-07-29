import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class MultiDFeature:
    indices: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        if len(self.indices) != len(self.values):
            raise ValueError(f'Size of feature indices and values do not match '
                             f'({len(self.indices)} vs {len(self.values)})')


@dataclass
class EntityFeatures:
    tabular_feature_names: list[str] = field(default_factory=list)
    tabular_features: pd.DataFrame = None
    multidimensional_features: dict[str, MultiDFeature] = field(default_factory=dict)

    def __post_init__(self):
        for tf in self.tabular_feature_names:
            if tf not in self.tabular_features.columns:
                raise ValueError(f'Tabular feature "{tf}" is specified, but missing in the feature table.')


@dataclass
class RawDataset:
    interactions: pd.DataFrame
    user_features: EntityFeatures
    item_features: EntityFeatures


@dataclass
class SplitData(RawDataset):
    user_indices: np.ndarray
    item_indices: np.ndarray


@dataclass
class AllSplitsData:
    tr_data: SplitData
    vd_data: SplitData
    te_data: SplitData
