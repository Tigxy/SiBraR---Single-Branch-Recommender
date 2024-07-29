import enum
import sys
import param
from typing import Optional
from enum import StrEnum, auto, Enum
from dataclasses import dataclass, field

from data.base_config_classes import excludable_field, BaseConfig, SoftBaseConfig


class DatasetSplitType(StrEnum):
    Random = auto()
    Temporal = auto()
    ColdStartUser = 'cold_start_user'
    ColdStartItem = 'cold_start_item'
    ColdStartBoth = 'cold_start_both'


class FeatureType(StrEnum):
    # e.g., gender (a single category per sample, could be one-hot encoded)
    CATEGORICAL = auto()
    # e.g., genre (multiple categories per sample, could be multi-hot encoded)
    TAG = auto()
    # e.g., age
    DISCRETE = auto()
    # e.g., preference
    CONTINUOUS = auto()
    # e.g., string representations of embeddings like "[1,2,3,4,5]" -> [1,2,3,4,5]
    # (this is how RecBole stores content data in dataframes)
    SEQUENCE = auto()
    # e.g., embeddings, like [1.,2.,3.,4.,5.]
    VECTOR = auto()
    # e.g., mel spectrogram , cover images, ...
    MATRIX = auto()


class ProcessingType(StrEnum):
    NONE = auto()
    ONE_HOT = auto()
    MULTI_HOT = auto()


class FeatureSamplingStrategy(Enum):
    UseAll = 0
    Alternate = 1
    SingleRandom = 2


class DatasetsEnum(StrEnum):
    """
    Enum to keep track of all the dataset available. Note that the name of the dataset  should correspond to a folder
    in data. e.g. ml-1m has a corresponding folder in /data
    """
    ml100k = auto()
    ml1m = auto()
    ml10m = auto()
    amazonvid2018 = auto()
    lfm2b2020 = auto()
    deliveryherosg = auto()
    onion = auto()
    onion18 = auto()
    onion18g = auto()
    kuai = auto()
    amazonvid2024 = auto()


class AlgorithmsEnum(StrEnum):
    uknn = auto()
    iknn = auto()
    ifknn = auto()
    mf = auto()
    ifeatmf = auto()
    sgdbias = auto()
    pop = auto()
    rand = auto()
    rbmf = auto()
    uprotomf = auto()
    iprotomf = auto()
    uiprotomf = auto()
    acf = auto()
    svd = auto()
    als = auto()
    p3alpha = auto()
    ease = auto()
    slim = auto()
    uprotomfs = auto()
    iprotomfs = auto()
    uiprotomfs = auto()
    ecf = auto()
    dmf = auto()
    dropoutnet = auto()
    sbnet = auto()
    ufeatmf = auto()


@dataclass
class DataLoaderConfig(BaseConfig):
    batch_size: int = 2
    shuffle: bool = False
    num_workers: int = 2
    prefetch_factor: int | None = 8
    persistent_workers: bool = True


@dataclass
class FeatureBaseDefinition(BaseConfig):
    name: str
    type: FeatureType


@dataclass
class FeatureDefinition(FeatureBaseDefinition):
    preprocessing: Optional[ProcessingType] = ProcessingType.NONE
    tag_split_sep: str = None


@dataclass
class RecDatasetConfig(SoftBaseConfig):
    name: str = excludable_field(None)
    split_set: str = 'train'
    dataset_path: str = None
    data_path: str = excludable_field(None)
    user_feature_definitions: list[FeatureDefinition] = None
    item_feature_definitions: list[FeatureDefinition] = None
    model_requires_train_interactions: bool = False
    keep_history_in_memory: bool = False


@dataclass
class InteractionDatasetConfig(RecDatasetConfig):
    model_requires_item_interactions: bool = False


@dataclass
class TrainDatasetConfig(InteractionDatasetConfig):
    n_negative_samples: int = 4
    use_dataset_negative_sampler: bool = True
    negative_sampling_strategy: str = 'uniform'  # 'popular' or 'uniform'
    sampling_popularity_squashing_factor: float = 1.  # only used for strategy 'popular'
    model_requires_pop_distribution: bool = False


@dataclass
class TrainUserRecDatasetConfig(TrainDatasetConfig):
    n_pos: int = 10


@dataclass
class FeatureTrainRecDatasetConfig(TrainDatasetConfig):
    feature_sampling_strategy: FeatureSamplingStrategy = FeatureSamplingStrategy.UseAll


@dataclass
class RunSettings(BaseConfig, param.Parameterized):
    seed: int = 42
    ray_verbose: bool = 1
    batch_verbose: bool = False
    in_tune: bool = False
    device: str = param.Selector(default='cpu', objects=('cpu', 'cuda'))


@dataclass
class WandBSettings(BaseConfig):
    use_wandb: bool = True
    wandb_path: str = 'wandb'

    # parameters for hyperparameter search with W&B
    sweep_id: str = None
    keep_top_runs: bool = 5


@dataclass
class LearningConfig(SoftBaseConfig, param.Parameterized):
    n_epochs: int = param.Integer(50, bounds=(1, None))
    max_batches_per_epoch: int = None
    lr: float = param.Number(1e-3, bounds=(1e-9, None))
    wd: float = param.Number(0, bounds=(0, None))
    optimizer: str = param.Selector(default='adam', objects=('adam', 'adagrad', 'adamw'))
    optimizing_metric: str = 'ndcg@10'
    rec_loss: str = param.Selector(default='bce', objects=('bce', 'bpr', 'sampled_softmax'))
    loss_aggregator: str = param.Selector(default='mean', objects=('mean', 'sum'))
    max_patience: int = param.Integer(sys.maxsize, bounds=(1, None))


@dataclass
class EvalConfig(BaseConfig):
    top_k: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 50, 100])
    metrics: list[str] = field(default_factory=lambda: ['ndcg', 'precision', 'recall',
                                                        'f_score', 'hitrate', 'coverage'])

    calculate_std: bool = True

    # whether to calculate metrics for the groups
    calculate_group_metrics: bool = False

    # list of user groups (features) for which to calculate the metrics.
    # Defaults to None to calculate metrics for all categorical user features
    user_group_features: list[str] = None


@dataclass
class ExperimentConfig(BaseConfig):
    run_id: str

    algorithm_type: AlgorithmsEnum
    algorithm_name: str = field(init=False)

    dataset_type: DatasetsEnum
    dataset_name: str = field(init=False)

    split_type: DatasetSplitType
    split_name: str = field(init=False)

    train_loader: DataLoaderConfig
    val_loader: DataLoaderConfig

    run_settings: RunSettings
    wandb: WandBSettings
    results_path: str

    # keep dataset and model as dictionary, as their class depends on the algorithm
    dataset: dict

    # evaluation config that can optionally be supplied
    eval: EvalConfig = field(default_factory=EvalConfig)

    # in case training should also be evaluated, specify this via its own EvalConfig
    train_eval: EvalConfig = None

    # optional base configuration, such that not all parameters have
    # to be specified in each new config
    base_configs: str | list['str'] = None

    # not all algorithms need parameters
    model: dict = field(default_factory=dict)

    # ... neither might they need parameters for fitting
    learn: LearningConfig = None

    # whether to do profiling and display output
    # note that this is not intended for use in actual training, thus
    # the number of training batches are limited
    profile_training: bool = False

    def __post_init__(self):
        self.algorithm_name = self.algorithm_type.name.lower()
        self.dataset_name = self.dataset_type.name.lower()
        self.split_name = self.split_type.name.lower()
