from typing import List
from enum import StrEnum, auto
from dataclasses import dataclass, field
from mashumaro.mixins.yaml import DataClassYAMLMixin

from data.config_classes import FeatureBaseDefinition


class SplitType(StrEnum):
    """
    Which kind of splitting to perform
    """
    Temporal = auto()
    ColdStart = auto()
    Random = auto()


class ColdStartType(StrEnum):
    """
    Which kind of cold start scenario should be run
    """
    User = auto()
    Item = auto()
    Both = auto()


@dataclass
class InteractionConfig(DataClassYAMLMixin):
    k_core: int = 5
    min_n_interactions: int = 2


@dataclass
class PreprocessingConfig(DataClassYAMLMixin):
    kind: str
    parameters: dict = field(default_factory=dict)


@dataclass
class NormalizationConfig(PreprocessingConfig):
    pass


@dataclass
class FeatureConfig(DataClassYAMLMixin, FeatureBaseDefinition):
    preprocessing: List[PreprocessingConfig] | None = field(default_factory=list)
    normalization: List[NormalizationConfig] | None = field(default_factory=list)

    def __post_init__(self):
        # per default, set attributes to empty lists.
        # while convenient, not sure whether this is the clearner than 'None'-checking in code
        self.preprocessing = self.preprocessing or []
        self.normalization = self.normalization or []


@dataclass
class SplitConfig(DataClassYAMLMixin):
    ratios: tuple
    split_type: SplitType

    # optional, as they are only used in some of the splitting types
    cold_start_type: ColdStartType = None
    seed: int = None


@dataclass
class DataPreprocessingConfig(DataClassYAMLMixin):
    split: SplitConfig
    interactions: InteractionConfig
    user_features: List[FeatureConfig] = field(default_factory=list)
    item_features: List[FeatureConfig] = field(default_factory=list)
