from enum import StrEnum, auto, Enum
from dataclasses import dataclass, field
from typing import List

import mashumaro

from data.base_config_classes import BaseConfig


class DropoutNetSamplingStrategy(Enum):
    Normal = auto()
    NoPreference = auto()

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


@dataclass
class FeatureModuleConfig(BaseConfig):
    feature_name: str
    embedding_dim: int
    pre_embedding_layers: List[int] = None
    post_embedding_layers: List[int] = None
    activation_fn: str = 'relu'


@dataclass
class DropoutNetEntityConfig(BaseConfig):
    features: List[FeatureModuleConfig]

    preference_layers: List[int]  # number of items will be prepended automatically
    common_hidden_layers: List[int]  # feature and preference dim, as well as shared common dim is added automatically
    activation_fn: str = 'relu'


@dataclass
class DropoutNetConfig(BaseConfig):
    user: DropoutNetEntityConfig
    item: DropoutNetEntityConfig
    shared_common_dim: int
    sampling_seed: int = 42


@dataclass
class SingleBranchFeatureConfig(BaseConfig):
    feature_name: str
    feature_hidden_layers: List[int] = None


class EmbeddingRegularizationType(Enum):
    """
    How to compare and regularize the embeddings of different modalities with one another
    """

    # no regularization
    NoRegularization = 'no_regularization'

    # select two of the available modalities and compare their embeddings
    PairwiseSingle = 'pairwise_single'

    # all other available modalities are compared to regularized w.r.t. a single modality
    # central modality has to be defined separately
    # per item in the batch, only central modality and one other modality is selected
    CentralModality = 'central_modality'

    # generate pairs between all the available modalities and compute their pairwise regularization
    # PairwiseFull = auto()
    # CentralModalityFull = auto()

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


@dataclass
class SingleBranchNetEntityConfig(BaseConfig):
    features: List[SingleBranchFeatureConfig]
    single_branch_hidden_layers: list[int]  # common modality size will be prepended automatically
    preference_hidden_layers: list[int]  # number of items will be prepended automatically
    common_modality_dim: int
    activation_fn: str = 'relu'
    train_modalities: set[str] = None
    eval_modalities: set[str] = None
    sampling_seed: int = 42
    single_branch_input_dropout: float = None
    aggregation_fn: str = 'mean'
    normalize_single_branch_input: bool = False
    embedding_regularization_type: EmbeddingRegularizationType = EmbeddingRegularizationType.NoRegularization
    central_modality: str = None  # only relevant for EmbeddingRegularizationType.CentralModality
    regularization_temperature: float = 1.
    regularization_weight: float = 1.
    apply_output_activation: bool = False
    apply_batch_normalization: bool = True
    apply_batch_norm_every: int = 0


@dataclass
class SingleBranchNetConfig(BaseConfig):
    user: SingleBranchNetEntityConfig | FeatureModuleConfig = field(
        metadata={
            "deserialize": lambda d: SingleBranchNetConfig._conditional_parse_entity_conf(d)
        }
    )

    item: SingleBranchNetEntityConfig | FeatureModuleConfig = field(
        metadata={
            "deserialize": lambda d: SingleBranchNetConfig._conditional_parse_entity_conf(d)
        }
    )

    shared_common_dim: int

    @classmethod
    def _conditional_parse_entity_conf(cls, conf: dict):
        try:
            return FeatureModuleConfig.from_dict(conf)
        except mashumaro.exceptions.MissingField:
            return SingleBranchNetEntityConfig.from_dict(conf)

    @property
    def is_user_sb_module(self) -> bool:
        return isinstance(self.user, SingleBranchNetEntityConfig)

    @property
    def is_item_sb_module(self) -> bool:
        return isinstance(self.item, SingleBranchNetEntityConfig)
