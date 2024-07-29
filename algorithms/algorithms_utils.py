from algorithms.graph_algs import P3alpha
from algorithms.knn_algs import UserKNN, ItemKNN, ItemFeatureKNN
from algorithms.linear_algs import EASE, SLIM
from algorithms.mf_algs import SVDAlgorithm, AlternatingLeastSquare, RBMF
from algorithms.naive_algs import PopularItems, RandomItems
from algorithms.sgd_alg import SGDMatrixFactorization, UProtoMF, UIProtoMF, IProtoMF, SGDBaseline, ACF, UProtoMFs, \
    IProtoMFs, UIProtoMFs, ECF, DeepMatrixFactorization, ItemFeatureMatrixFactorization, DropoutNet, SingleBranchNet, \
    UserFeatureMatrixFactorization
from data.config_classes import AlgorithmsEnum

enum_to_value = {
    AlgorithmsEnum.uknn: UserKNN,
    AlgorithmsEnum.iknn: ItemKNN,
    AlgorithmsEnum.ifknn: ItemFeatureKNN,
    AlgorithmsEnum.mf: SGDMatrixFactorization,
    AlgorithmsEnum.ifeatmf: ItemFeatureMatrixFactorization,
    AlgorithmsEnum.sgdbias: SGDBaseline,
    AlgorithmsEnum.pop: PopularItems,
    AlgorithmsEnum.rand: RandomItems,
    AlgorithmsEnum.rbmf: RBMF,
    AlgorithmsEnum.uprotomf: UProtoMF,
    AlgorithmsEnum.iprotomf: IProtoMF,
    AlgorithmsEnum.uiprotomf: UIProtoMF,
    AlgorithmsEnum.acf: ACF,
    AlgorithmsEnum.svd: SVDAlgorithm,
    AlgorithmsEnum.als: AlternatingLeastSquare,
    AlgorithmsEnum.p3alpha: P3alpha,
    AlgorithmsEnum.ease: EASE,
    AlgorithmsEnum.slim: SLIM,
    AlgorithmsEnum.uprotomfs: UProtoMFs,
    AlgorithmsEnum.iprotomfs: IProtoMFs,
    AlgorithmsEnum.uiprotomfs: UIProtoMFs,
    AlgorithmsEnum.ecf: ECF,
    AlgorithmsEnum.dmf: DeepMatrixFactorization,
    AlgorithmsEnum.dropoutnet: DropoutNet,
    AlgorithmsEnum.sbnet: SingleBranchNet,
    AlgorithmsEnum.ufeatmf: UserFeatureMatrixFactorization
}


def get_algorithm_class(alg: AlgorithmsEnum):
    return enum_to_value[alg]
