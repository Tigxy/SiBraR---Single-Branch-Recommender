import os
import re
import math
import inspect
import logging
import numpy as np
from scipy import sparse as sp
from typing import Union, Tuple, Dict, List

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data

from algorithms.base_classes import SGDBasedRecommenderAlgorithm, RecommenderAlgorithm, PrototypeWrapper
from data.Feature import Feature
from data.config_classes import FeatureType, FeatureDefinition
from data.dataset import RecDataset, TrainRecDataset, InteractionRecDataset
from data.module_config_classes import DropoutNetConfig, DropoutNetSamplingStrategy, FeatureModuleConfig, \
    DropoutNetEntityConfig, SingleBranchNetEntityConfig, SingleBranchNetConfig, EmbeddingRegularizationType
from explanations.utils import protomfs_post_val, protomf_post_val_light
from modules.polylinear import PolyLinear
from train.regularization_losses import InfoNCE, ZeroLossModule
from train.utils import general_weight_init
from utilities.utils import row_wise_sample

AGGREGATION_FUNCTIONS = {
    'mean': torch.mean,
    # max returns (values, indices), thus wrap it to only return the values
    'max': lambda *args, **kwargs: torch.max(*args, **kwargs).values
}


def compute_norm_cosine_sim(x: torch.Tensor, y: torch.Tensor):
    """
    Computes the normalized shifted cosine similarity between two tensors.
    x and y have the same last dimension.
    """
    x_norm = F.normalize(x)
    y_norm = F.normalize(y)

    sim_mtx = (1 + x_norm @ y_norm.T) / 2
    sim_mtx = torch.clamp(sim_mtx, min=0., max=1.)

    return sim_mtx


def compute_shifted_cosine_sim(x: torch.Tensor, y: torch.Tensor):
    """
    Computes the shifted cosine similarity between two tensors.
    x and y have the same last dimension.
    """
    x_norm = F.normalize(x)
    y_norm = F.normalize(y)

    sim_mtx = (1 + x_norm @ y_norm.T)
    sim_mtx = torch.clamp(sim_mtx, min=0., max=2.)

    return sim_mtx


def compute_cosine_sim(x: torch.Tensor, y: torch.Tensor):
    """
    Computes the cosine similarity between two tensors.
    x and y have the same last dimension.
    """
    x_norm = F.normalize(x)
    y_norm = F.normalize(y)

    sim_mtx = x_norm @ y_norm.T
    sim_mtx = torch.clamp(sim_mtx, min=-1., max=1.)

    return sim_mtx


def entropy_from_softmax(p: torch.Tensor, p_unnorm: torch.Tensor):
    """
    Computes the entropy of a probability distribution assuming the distribution was obtained by softmax. It uses the
    un-normalized probabilities for numerical stability.
    @param p: tensor containing the probability of events xs. Shape is [*, n_events]
    @param p_unnorm: tensor contained the un-normalized probabilities (logits) of events xs. Shape is [*, n_events]
    @return: entropy of p. Shape is [*]
    """

    return (- (p * (p_unnorm - torch.logsumexp(p_unnorm, dim=-1, keepdim=True)))).sum(-1)


class SGDBaseline(SGDBasedRecommenderAlgorithm):
    """
    Implements a simple baseline comprised of biases (global, user, and item).
    See https://dl.acm.org/doi/10.1145/1401890.1401944
    """

    def __init__(self, n_users: int, n_items: int):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items

        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.apply(general_weight_init)

        self.name = 'SGDBaseline'

        logging.info(f'Built {self.name} module\n')

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.user_bias(u_idxs)

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.item_bias(i_idxs).squeeze()

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        out = u_repr + i_repr + self.global_bias
        return out

    @staticmethod
    def build_from_conf(conf: dict, dataset):
        return SGDBaseline(dataset.n_users, dataset.n_items)


class SGDMatrixFactorization(SGDBasedRecommenderAlgorithm):
    """
    Implements a simple Matrix Factorization model trained with gradient descent
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, use_user_bias: bool = False,
                 use_item_bias: bool = False, use_global_bias: bool = False):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.use_user_bias = use_user_bias
        self.use_item_bias = use_item_bias
        self.use_global_bias = use_global_bias

        self.user_embeddings = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embeddings = nn.Embedding(self.n_items, self.embedding_dim)

        if self.use_user_bias:
            self.user_bias = nn.Embedding(self.n_users, 1)
        if self.use_item_bias:
            self.item_bias = nn.Embedding(self.n_items, 1)

        self.apply(general_weight_init)

        if self.use_global_bias:
            self.global_bias = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.name = 'SGDMatrixFactorization'

        logging.info(f'Built {self.name} module\n'
                     f'- embedding_dim: {self.embedding_dim} \n'
                     f'- use_user_bias: {self.use_user_bias} \n'
                     f'- use_item_bias: {self.use_item_bias} \n'
                     f'- use_global_bias: {self.use_global_bias}')

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if self.use_user_bias:
            return self.user_embeddings(u_idxs), self.user_bias(u_idxs)
        else:
            return self.user_embeddings(u_idxs)

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if self.use_item_bias:
            return self.item_embeddings(i_idxs), self.item_bias(i_idxs).squeeze()
        return self.item_embeddings(i_idxs)

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        if isinstance(u_repr, tuple):
            u_embed, u_bias = u_repr
        else:
            u_embed = u_repr

        if isinstance(i_repr, tuple):
            i_embed, i_bias = i_repr
        else:
            i_embed = i_repr

        out = (u_embed[:, None, :] * i_embed).sum(dim=-1)

        if self.use_user_bias:
            out += u_bias[:, None]
        if self.use_item_bias:
            out += i_bias
        if self.use_global_bias:
            out += self.global_bias
        return out

    @staticmethod
    def build_from_conf(conf: dict, dataset):
        return SGDMatrixFactorization(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['use_user_bias'],
                                      conf['use_item_bias'], conf['use_global_bias'])


class ACF(PrototypeWrapper):
    """
    Implements Anchor-based Collaborative Filtering by Barkan et al.(https://dl.acm.org/doi/pdf/10.1145/3459637.3482056)
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, n_anchors: int = 20,
                 delta_exc: float = 1e-1, delta_inc: float = 1e-2):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_anchors = n_anchors
        self.delta_exc = delta_exc
        self.delta_inc = delta_inc

        # NB. In order to ensure stability, ACF's weights **need** not to be initialized with small values.
        self.anchors = nn.Parameter(torch.randn([self.n_anchors, self.embedding_dim]), requires_grad=True)

        self.user_embed = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embed = nn.Embedding(self.n_items, self.embedding_dim)

        self._acc_exc = 0
        self._acc_inc = 0

        self.name = 'ACF'

        logging.info(f'Built {self.name} model \n'
                     f'- n_users: {self.n_users} \n'
                     f'- n_items: {self.n_items} \n'
                     f'- embedding_dim: {self.embedding_dim} \n'
                     f'- n_anchors: {self.n_anchors} \n'
                     f'- delta_exc: {self.delta_exc} \n'
                     f'- delta_inc: {self.delta_inc} \n')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_repr = self.get_user_representations(u_idxs)
        i_repr = self.get_item_representations(i_idxs)

        dots = self.combine_user_item_representations(u_repr, i_repr)

        _, c_i, c_i_unnorm = i_repr

        # Exclusiveness constraint
        exc_values = entropy_from_softmax(c_i, c_i_unnorm)  # [batch_size, n_neg +1] or [batch_size]
        exc_loss = exc_values.mean()

        # Inclusiveness constraint
        c_i_flat = c_i.reshape(-1, self.n_anchors)  # [*, n_anchors]
        q_k = c_i_flat.sum(dim=0) / c_i.sum()  # [n_anchors]
        inc_entropy = (- q_k * torch.log(q_k)).sum()
        inc_loss = math.log(self.n_anchors) - inc_entropy  # Maximizing the Entropy

        self._acc_exc += exc_loss
        self._acc_inc += inc_loss

        return dots

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_embed = self.user_embed(u_idxs)  # [batch_size, embedding_dim]
        c_u = u_embed @ self.anchors.T  # [batch_size, n_anchors]
        c_u = nn.Softmax(dim=-1)(c_u)

        u_anc = c_u @ self.anchors  # [batch_size, embedding_dim]

        return u_anc

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_embed = self.item_embed(i_idxs)  # [batch_size, (n_neg + 1), embedding_dim]
        c_i_unnorm = i_embed @ self.anchors.T  # [batch_size, (n_neg + 1), n_anchors]
        c_i = nn.Softmax(dim=-1)(c_i_unnorm)  # [batch_size, (n_neg + 1), n_anchors]

        i_anc = c_i @ self.anchors  # [batch_size, (n_neg + 1), embedding_dim]
        return i_anc, c_i, c_i_unnorm

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        u_anc = u_repr
        i_anc = i_repr[0]
        dots = (u_anc.unsqueeze(-2) * i_anc).sum(dim=-1)
        return dots

    def get_item_representations_pre_tune(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_embed = self.item_embed(i_idxs)  # [batch_size, (n_neg + 1), embedding_dim]
        c_i_unnorm = i_embed @ self.anchors.T  # [batch_size, (n_neg + 1), n_anchors]
        c_i = nn.Softmax(dim=-1)(c_i_unnorm)  # [batch_size, (n_neg + 1), n_anchors]
        return c_i

    def get_item_representations_post_tune(self, c_i: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_anc = c_i @ self.anchors  # [batch_size, (n_neg + 1), embedding_dim]
        return i_anc, c_i, None

    def get_user_representations_pre_tune(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_embed = self.user_embed(u_idxs)  # [batch_size, embedding_dim]
        c_u = u_embed @ self.anchors.T  # [batch_size, n_anchors]
        c_u = nn.Softmax(dim=-1)(c_u)
        return c_u

    def get_user_representations_post_tune(self, c_u: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_anc = c_u @ self.anchors  # [batch_size, embedding_dim]
        return u_anc

    def get_and_reset_other_loss(self) -> Dict:
        _acc_exc, _acc_inc = self._acc_exc, self._acc_inc
        self._acc_exc = self._acc_inc = 0
        exc_loss = self.delta_exc * _acc_exc
        inc_loss = self.delta_inc * _acc_inc

        return {
            'reg_loss': exc_loss + inc_loss,
            'exc_loss': exc_loss,
            'inc_loss': inc_loss
        }

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return ACF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_anchors'],
                   conf['delta_exc'], conf['delta_inc'])

    def post_val(self, curr_epoch: int):
        return protomf_post_val_light(
            self.anchors,
            self.item_embed.weight,
            compute_cosine_sim,
            lambda x: 1 - x,
            "Items",
            curr_epoch)


class UProtoMF(PrototypeWrapper):
    """
    Implements the ProtoMF model with user prototypes as defined in https://dl.acm.org/doi/abs/10.1145/3523227.3546756
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, n_prototypes: int = 20,
                 sim_proto_weight: float = 1., sim_batch_weight: float = 1.):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes
        self.sim_proto_weight = sim_proto_weight
        self.sim_batch_weight = sim_batch_weight

        self.user_embed = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embed = nn.Embedding(self.n_items, self.n_prototypes)

        self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]) * .1 / self.embedding_dim,
                                       requires_grad=True)

        self.user_embed.apply(general_weight_init)
        self.item_embed.apply(general_weight_init)

        self._acc_r_proto = 0
        self._acc_r_batch = 0

        self.name = 'UProtoMF'

        logging.info(f'Built {self.name} model \n'
                     f'- n_users: {self.n_users} \n'
                     f'- n_items: {self.n_items} \n'
                     f'- embedding_dim: {self.embedding_dim} \n'
                     f'- n_prototypes: {self.n_prototypes} \n'
                     f'- sim_proto_weight: {self.sim_proto_weight} \n'
                     f'- sim_batch_weight: {self.sim_batch_weight} \n')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_repr = self.get_user_representations(u_idxs)
        i_repr = self.get_item_representations(i_idxs)

        dots = self.combine_user_item_representations(u_repr, i_repr)

        self.compute_reg_losses(u_repr)

        return dots

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_embed = self.user_embed(u_idxs)
        sim_mtx = compute_shifted_cosine_sim(u_embed, self.prototypes)  # [batch_size, n_prototypes]

        return sim_mtx

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.item_embed(i_idxs)

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        dots = (u_repr.unsqueeze(-2) * i_repr).sum(dim=-1)
        return dots

    def compute_reg_losses(self, sim_mtx):
        # Compute regularization losses
        sim_mtx = sim_mtx.reshape(-1, self.n_prototypes)
        dis_mtx = (2 - sim_mtx)  # Equivalent to maximizing the similarity.
        self._acc_r_proto += dis_mtx.min(dim=0).values.mean()
        self._acc_r_batch += dis_mtx.min(dim=1).values.mean()

    def get_and_reset_other_loss(self) -> Dict:
        acc_r_proto, acc_r_batch = self._acc_r_proto, self._acc_r_batch
        self._acc_r_proto = self._acc_r_batch = 0
        proto_loss = self.sim_proto_weight * acc_r_proto
        batch_loss = self.sim_batch_weight * acc_r_batch
        return {
            'reg_loss': proto_loss + batch_loss,
            'proto_loss': proto_loss,
            'batch_loss': batch_loss
        }

    def get_user_representations_pre_tune(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_repr = self.get_user_representations(u_idxs)
        return u_repr

    def get_user_representations_post_tune(self, u_repr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return u_repr

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return UProtoMF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_prototypes'],
                        conf['sim_proto_weight'], conf['sim_batch_weight'])

    def post_val(self, curr_epoch: int):
        return protomf_post_val_light(
            self.prototypes,
            self.user_embed.weight,
            compute_shifted_cosine_sim,
            lambda x: 2 - x,
            "Users",
            curr_epoch)


class IProtoMF(PrototypeWrapper):
    """
    Implements the ProtoMF model with item prototypes as defined in https://dl.acm.org/doi/abs/10.1145/3523227.3546756
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, n_prototypes: int = 20,
                 sim_proto_weight: float = 1., sim_batch_weight: float = 1.):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes
        self.sim_proto_weight = sim_proto_weight
        self.sim_batch_weight = sim_batch_weight

        self.user_embed = nn.Embedding(self.n_users, self.n_prototypes)
        self.item_embed = nn.Embedding(self.n_items, self.embedding_dim)

        self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]) * .1 / self.embedding_dim,
                                       requires_grad=True)

        self.user_embed.apply(general_weight_init)
        self.item_embed.apply(general_weight_init)

        self._acc_r_proto = 0
        self._acc_r_batch = 0

        self.name = 'IProtoMF'

        logging.info(f'Built {self.name} model \n'
                     f'- n_users: {self.n_users} \n'
                     f'- n_items: {self.n_items} \n'
                     f'- embedding_dim: {self.embedding_dim} \n'
                     f'- n_prototypes: {self.n_prototypes} \n'
                     f'- sim_proto_weight: {self.sim_proto_weight} \n'
                     f'- sim_batch_weight: {self.sim_batch_weight} \n')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_repr = self.get_user_representations(u_idxs)
        i_repr = self.get_item_representations(i_idxs)

        dots = self.combine_user_item_representations(u_repr, i_repr)

        self.compute_reg_losses(i_repr)

        return dots

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.user_embed(u_idxs)

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_embed = self.item_embed(i_idxs)
        i_embed = i_embed.reshape(-1, i_embed.shape[-1])
        sim_mtx = compute_shifted_cosine_sim(i_embed, self.prototypes)
        sim_mtx = sim_mtx.reshape(list(i_idxs.shape) + [self.n_prototypes])

        return sim_mtx

    def get_item_representations_pre_tune(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_repr = self.get_item_representations(i_idxs)
        return i_repr

    def get_item_representations_post_tune(self, i_repr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return i_repr

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        dots = (u_repr.unsqueeze(-2) * i_repr).sum(dim=-1)
        return dots

    def compute_reg_losses(self, sim_mtx):
        # Compute regularization losses
        sim_mtx = sim_mtx.reshape(-1, self.n_prototypes)
        dis_mtx = (2 - sim_mtx)  # Equivalent to maximizing the similarity.
        self._acc_r_proto += dis_mtx.min(dim=0).values.mean()
        self._acc_r_batch += dis_mtx.min(dim=1).values.mean()

    def get_and_reset_other_loss(self) -> Dict:
        acc_r_proto, acc_r_batch = self._acc_r_proto, self._acc_r_batch
        self._acc_r_proto = self._acc_r_batch = 0
        proto_loss = self.sim_proto_weight * acc_r_proto
        batch_loss = self.sim_batch_weight * acc_r_batch
        return {
            'reg_loss': proto_loss + batch_loss,
            'proto_loss': proto_loss,
            'batch_loss': batch_loss
        }

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return IProtoMF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_prototypes'],
                        conf['sim_proto_weight'], conf['sim_batch_weight'])

    def post_val(self, curr_epoch: int):
        return protomf_post_val_light(
            self.prototypes,
            self.item_embed.weight,
            compute_shifted_cosine_sim,
            lambda x: 2 - x,
            "Items",
            curr_epoch)


class UIProtoMF(PrototypeWrapper):
    """
    Implements the ProtoMF model with item and user prototypes as defined in https://dl.acm.org/doi/abs/10.1145/3523227.3546756
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, u_n_prototypes: int = 20,
                 i_n_prototypes: int = 20, u_sim_proto_weight: float = 1., u_sim_batch_weight: float = 1.,
                 i_sim_proto_weight: float = 1., i_sim_batch_weight: float = 1.):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.uprotomf = UProtoMF(n_users, n_items, embedding_dim, u_n_prototypes,
                                 u_sim_proto_weight, u_sim_batch_weight)

        self.iprotomf = IProtoMF(n_users, n_items, embedding_dim, i_n_prototypes,
                                 i_sim_proto_weight, i_sim_batch_weight)

        self.u_to_i_proj = nn.Linear(self.embedding_dim, i_n_prototypes, bias=False)  # UProtoMF -> IProtoMF
        self.i_to_u_proj = nn.Linear(self.embedding_dim, u_n_prototypes, bias=False)  # IProtoMF -> UProtoMF

        self.u_to_i_proj.apply(general_weight_init)
        self.i_to_u_proj.apply(general_weight_init)

        # Deleting unused parameters

        del self.uprotomf.item_embed
        del self.iprotomf.user_embed

        self.name = 'UIProtoMF'

        logging.info(f'Built {self.name} model \n')

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_sim_mtx = self.uprotomf.get_user_representations(u_idxs)
        u_proj = self.u_to_i_proj(self.uprotomf.user_embed(u_idxs))

        return u_sim_mtx, u_proj

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_sim_mtx = self.iprotomf.get_item_representations(i_idxs)
        i_proj = self.i_to_u_proj(self.iprotomf.item_embed(i_idxs))

        return i_sim_mtx, i_proj

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        u_sim_mtx, u_proj = u_repr
        i_sim_mtx, i_proj = i_repr

        u_dots = (u_sim_mtx.unsqueeze(-2) * i_proj).sum(dim=-1)
        i_dots = (u_proj.unsqueeze(-2) * i_sim_mtx).sum(dim=-1)
        dots = u_dots + i_dots
        return dots

    def get_item_representations_pre_tune(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_repr = self.get_item_representations(i_idxs)
        return i_repr

    def get_item_representations_post_tune(self, i_repr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return i_repr

    def get_user_representations_pre_tune(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_repr = self.get_user_representations(u_idxs)
        return u_repr

    def get_user_representations_post_tune(self, u_repr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return u_repr

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_repr = self.get_user_representations(u_idxs)
        i_repr = self.get_item_representations(i_idxs)

        dots = self.combine_user_item_representations(u_repr, i_repr)

        u_sim_mtx, _ = u_repr
        i_sim_mtx, _ = i_repr
        self.uprotomf.compute_reg_losses(u_sim_mtx)
        self.iprotomf.compute_reg_losses(i_sim_mtx)

        return dots

    def get_and_reset_other_loss(self) -> Dict:
        u_reg = {'user_' + k: v for k, v in self.uprotomf.get_and_reset_other_loss().items()}
        i_reg = {'item_' + k: v for k, v in self.iprotomf.get_and_reset_other_loss().items()}
        return {
            'reg_loss': u_reg.pop('user_reg_loss') + i_reg.pop('item_reg_loss'),
            **u_reg,
            **i_reg
        }

    def post_val(self, curr_epoch: int):
        uprotomf_post_val = {'user_' + k: v for k, v in self.uprotomf.post_val(curr_epoch).items()}
        iprotomf_post_val = {'item_' + k: v for k, v in self.iprotomf.post_val(curr_epoch).items()}
        return {**uprotomf_post_val, **iprotomf_post_val}

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return UIProtoMF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['u_n_prototypes'],
                         conf['i_n_prototypes'], conf['u_sim_proto_weight'], conf['u_sim_batch_weight'],
                         conf['i_sim_proto_weight'], conf['i_sim_batch_weight'])


class UProtoMFs(SGDBasedRecommenderAlgorithm):
    """
    Implements a slightly simplified ProtoMF model with user prototypes. It differs from the original ProtoMF on:
        - No regularization losses are enforced
        - Entity-to-Prototype similarities can be negative (full-cosine similarity).
        - Other-entity (in this case items) weights are constrained to be positive. (Using a RelU)
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, n_prototypes: int = 20):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes

        self.user_embed = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embed = nn.Embedding(self.n_items, self.n_prototypes)
        self.relu = nn.ReLU()
        self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]) * .1 / self.embedding_dim,
                                       requires_grad=True)

        self.user_embed.apply(general_weight_init)
        torch.nn.init.trunc_normal_(self.item_embed.weight, mean=0.5, std=.1 / self.embedding_dim, a=0, b=1)

        self.name = 'UProtoMFs'

        logging.info(f'Built {self.name} model \n'
                     f'- n_users: {self.n_users} \n'
                     f'- n_items: {self.n_items} \n'
                     f'- embedding_dim: {self.embedding_dim} \n'
                     f'- n_prototypes: {self.n_prototypes} \n')

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_embed = self.user_embed(u_idxs)
        sim_mtx = compute_cosine_sim(u_embed, self.prototypes)  # [batch_size, n_prototypes]

        return sim_mtx

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.relu(self.item_embed(i_idxs))  # [batch_size, n_neg + 1, n_prototypes]

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        dots = (u_repr.unsqueeze(-2) * i_repr).sum(dim=-1)
        return dots

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return UProtoMFs(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_prototypes'])

    def post_val(self, curr_epoch: int):
        return protomfs_post_val(
            self.prototypes,
            self.user_embed.weight,
            self.relu(self.item_embed.weight),
            compute_cosine_sim,
            lambda x: 1 - x,
            "Users",
            curr_epoch)


class IProtoMFs(SGDBasedRecommenderAlgorithm):
    """
    Implements a slightly simplified ProtoMF model with item prototypes. It differs from the original ProtoMF on:
        - No regularization losses are enforced
        - Entity-to-Prototype similarities can be negative (full-cosine similarity).
        - Other-entity (in this case items) weights are constrained to be positive. (Using a RelU)
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, n_prototypes: int = 20):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes

        self.user_embed = nn.Embedding(self.n_users, self.n_prototypes)
        self.item_embed = nn.Embedding(self.n_items, self.embedding_dim)
        self.relu = nn.ReLU()
        self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]) * .1 / self.embedding_dim,
                                       requires_grad=True)

        self.item_embed.apply(general_weight_init)
        torch.nn.init.trunc_normal_(self.user_embed.weight, mean=0.5, std=.1 / self.embedding_dim, a=0, b=1)

        self.name = 'IProtoMFs'

        logging.info(f'Built {self.name} model \n'
                     f'- n_users: {self.n_users} \n'
                     f'- n_items: {self.n_items} \n'
                     f'- embedding_dim: {self.embedding_dim} \n'
                     f'- n_prototypes: {self.n_prototypes} \n')

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.relu(self.user_embed(u_idxs))  # [batch_size, n_prototypes]

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_embed = self.item_embed(i_idxs)
        i_embed = i_embed.reshape(-1, i_embed.shape[-1])
        sim_mtx = compute_cosine_sim(i_embed, self.prototypes)
        sim_mtx = sim_mtx.reshape(list(i_idxs.shape) + [self.n_prototypes])
        return sim_mtx

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        dots = (u_repr.unsqueeze(-2) * i_repr).sum(dim=-1)
        return dots

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return IProtoMFs(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_prototypes'])

    def post_val(self, curr_epoch: int):
        return protomfs_post_val(
            self.prototypes,
            self.item_embed.weight,
            self.relu(self.user_embed.weight),
            compute_cosine_sim,
            lambda x: 1 - x,
            "Items",
            curr_epoch)


class UIProtoMFs(SGDBasedRecommenderAlgorithm):
    """
    Implements a slightly simplified ProtoMF model with user and item prototypes.
    It differs from the original ProtoMF on:
        - No regularization losses are enforced
        - Entity-to-Prototype similarities can be negative (full-cosine similarity).
        - Other-entity (in this case items) weights are constrained to be positive. (Using a RelU)
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, u_n_prototypes: int = 20,
                 i_n_prototypes: int = 20):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.uprotomfs = UProtoMFs(n_users, n_items, embedding_dim, u_n_prototypes)

        self.iprotomfs = IProtoMFs(n_users, n_items, embedding_dim, i_n_prototypes)

        self.u_to_i_proj = nn.Linear(self.embedding_dim, i_n_prototypes, bias=False)  # UProtoMFs -> IProtoMFs
        self.i_to_u_proj = nn.Linear(self.embedding_dim, u_n_prototypes, bias=False)  # IProtoMFs -> UProtoMFs

        self.relu = nn.ReLU()

        self.u_to_i_proj.apply(general_weight_init)
        self.i_to_u_proj.apply(general_weight_init)

        # Deleting unused parameters

        del self.uprotomfs.item_embed
        del self.iprotomfs.user_embed

        self.name = 'UIProtoMFs'

        logging.info(f'Built {self.name} model \n')

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_sim_mtx = self.uprotomfs.get_user_representations(u_idxs)
        u_proj = self.relu(self.u_to_i_proj(self.uprotomfs.user_embed(u_idxs)))

        return u_sim_mtx, u_proj

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_sim_mtx = self.iprotomfs.get_item_representations(i_idxs)
        i_proj = self.relu(self.i_to_u_proj(self.iprotomfs.item_embed(i_idxs)))

        return i_sim_mtx, i_proj

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        u_sim_mtx, u_proj = u_repr
        i_sim_mtx, i_proj = i_repr

        u_dots = (u_sim_mtx.unsqueeze(-2) * i_proj).sum(dim=-1)
        i_dots = (u_proj.unsqueeze(-2) * i_sim_mtx).sum(dim=-1)
        dots = u_dots + i_dots
        return dots

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return UIProtoMFs(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['u_n_prototypes'],
                          conf['i_n_prototypes'])

    def post_val(self, curr_epoch: int):
        uprotomfs_post_val = protomfs_post_val(self.uprotomfs.prototypes,
                                               self.uprotomfs.user_embed.weight,
                                               self.relu(self.i_to_u_proj(self.iprotomfs.item_embed.weight)),
                                               compute_cosine_sim,
                                               lambda x: (1 - x) / 2,
                                               "Users",
                                               curr_epoch)
        iprotomfs_post_val = protomfs_post_val(self.iprotomfs.prototypes,
                                               self.iprotomfs.item_embed.weight,
                                               self.relu(self.u_to_i_proj(self.uprotomfs.user_embed.weight)),
                                               compute_cosine_sim,
                                               lambda x: (1 - x) / 2,
                                               "Items",
                                               curr_epoch)
        uprotomfs_post_val = {'user_' + k: v for k, v in uprotomfs_post_val.items()}
        iprotomfs_post_val = {'item_' + k: v for k, v in iprotomfs_post_val.items()}
        return {**uprotomfs_post_val, **iprotomfs_post_val}


class UIProtoMFsCombine(RecommenderAlgorithm):
    """
    It encases UProtoMFs and IProtoMFs. Make sure that the models weights are loaded before calling __init__.
    No optimization is needed.
    """

    def save_model_to_path(self, path: str):
        raise ValueError(
            'This class cannot be saved to path since it made of 2 separate models (that should have been already saved'
            ' somewhere). Save the UProtoMFs and IProtoMFs models separately. If you want to optimize a UIProtoMF model,'
            ' use the UIProtoMF/s classes.')

    def load_model_from_path(self, path: str):
        raise ValueError(
            'This class cannot be loaded from path since it made of 2 separate models (that should have been already loaded'
            'from somewhere). Load the UProtoMFs and IProtoMFs models separately. If you want to optimize a UIProtoMF model,'
            ' use the UIProtoMF/s classes.')

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        raise ValueError(
            'This class cannot be built from conf since it made of 2 separate models. If you want to optimize a UIProtoMF model,'
            ' use the UIProtoMF/s classes.')

    def __init__(self, uprotomfs: UProtoMFs, iprotomfs: IProtoMFs):
        super().__init__()

        self.uprotomfs = uprotomfs
        self.iprotomfs = iprotomfs

        self.name = 'UIProtoMFsCombine'

        logging.info(f'Built {self.name} model \n')

    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        return self.uprotomfs.predict(u_idxs, i_idxs) + self.iprotomfs.predict(u_idxs, i_idxs)


class ECF(PrototypeWrapper):
    """
    Implements the ECF model from https://dl.acm.org/doi/10.1145/3543507.3583303
    """

    def __init__(self, n_users: int, n_items: int, tag_matrix: sp.csr_matrix, interaction_matrix: sp.csr_matrix,
                 embedding_dim: int = 100, n_clusters: int = 64, top_n: int = 20, top_m: int = 20,
                 temp_masking: float = 2., temp_tags: float = 2., top_p: int = 4, lam_cf: float = 0.6,
                 lam_ind: float = 1., lam_ts: float
                 = 1.):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.tag_matrix = nn.Parameter(torch.from_numpy(tag_matrix.A), requires_grad=False).float()
        self.interaction_matrix = nn.Parameter(torch.from_numpy(interaction_matrix.A), requires_grad=False).float()

        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.top_n = top_n
        self.top_m = top_m
        self.temp_masking = temp_masking
        self.temp_tags = temp_tags
        self.top_p = top_p

        self.lam_cf = lam_cf
        self.lam_ind = lam_ind
        self.lam_ts = lam_ts

        self.user_embed = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embed = nn.Embedding(self.n_items, self.embedding_dim)

        indxs = torch.randperm(self.n_items)[:self.n_clusters]
        self.clusters = nn.Parameter(self.item_embed.weight[indxs].detach(), requires_grad=True)

        self._acc_ts = 0
        self._acc_ind = 0
        self._acc_cf = 0

        # Parameters are set every batch
        self._x_tildes = None
        self._xs = None

        self.name = 'ECF'

        logging.info(f'Built {self.name} model \n'
                     f'- n_users: {self.n_users} \n'
                     f'- n_items: {self.n_items} \n'
                     f'- embedding_dim: {self.embedding_dim} \n'
                     f'- n_clusters: {self.n_clusters} \n'
                     f'- lam_cf: {self.lam_cf} \n'
                     f'- top_n: {self.top_n} \n'
                     f'- top_m: {self.top_m} \n'
                     f'- temp_masking: {self.temp_masking} \n'
                     f'- temp_tags: {self.temp_tags} \n'
                     f'- top_p: {self.top_p} \n')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        i_repr = self.get_item_representations(i_idxs)
        # NB. item representations should be generated before calling user_representations
        u_repr = self.get_user_representations(u_idxs)

        dots = self.combine_user_item_representations(u_repr, i_repr)

        # Tag Loss
        # N.B. Frequency weighting factor is already included in the tag_matrix.
        d_c = self._xs.T @ self.tag_matrix.to(u_idxs.device)  # [n_clusters, n_tags]
        # since log is a monotonic function we can invert the order between log and topk
        log_b_c = nn.LogSoftmax(-1)(d_c / self.temp_tags)
        top_log_b_c = log_b_c.topk(self.top_p, dim=-1).values  # [n_clusters, top_p]

        loss_tags = (- top_log_b_c).sum()
        self._acc_ts += loss_tags

        # Independence Loss
        sim_mtx = compute_cosine_sim(self.clusters, self.clusters)
        self_sim = torch.diag(- nn.LogSoftmax(dim=-1)(sim_mtx))

        self._acc_ind += self_sim.sum()

        # BPR Loss
        u_embed = u_repr[1]
        i_embed = i_repr[1]

        logits = (u_embed.unsqueeze(-2) * i_embed).sum(dim=-1)

        pos_logits = logits[:, 0].unsqueeze(1)  # [batch_size,1]
        neg_logits = logits[:, 1:]  # [batch_size,n_neg]

        diff_logits = (pos_logits - neg_logits).flatten()
        labels = torch.ones_like(diff_logits, device=diff_logits.device)

        bpr_loss = nn.BCEWithLogitsLoss()(diff_logits, labels)
        self._acc_cf += bpr_loss

        return dots

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        y_u = self.interaction_matrix[u_idxs]  # [batch_size, n_items]
        u_embed = self.user_embed(u_idxs)

        a_tilde = y_u @ self._x_tildes  # [batch_size, n_clusters]

        # Creating exact mask
        m = torch.zeros_like(a_tilde).to(a_tilde.device)
        a_tilde_tops = a_tilde.topk(self.top_n).indices
        dummy_column = torch.arange(a_tilde.shape[0])[:, None].to(a_tilde.device)
        m[dummy_column, a_tilde_tops] = True

        # Creating approximated mask
        m_tilde = nn.Softmax(dim=-1)(a_tilde / self.temp_masking)

        # Putting together the masks
        m_hat = m_tilde + (m - m_tilde).detach()

        # Building affiliation vector
        a_i = nn.Sigmoid()(a_tilde) * m_hat

        return a_i, u_embed

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        self._generate_item_representations()

        i_embed = self.item_embed(i_idxs)  # [batch_size, embed_dim] or [batch_size, n_neg + 1, embed_dim]

        x_i = self._xs[i_idxs]

        return x_i, i_embed

    def _generate_item_representations(self):
        i_embed = self.item_embed.weight  # [n_items, embed_d]
        self._x_tildes = compute_cosine_sim(i_embed, self.clusters)  # [n_items, n_clusters]

        # Creating exact mask
        m = torch.zeros_like(self._x_tildes).to(self._x_tildes.device)
        x_tilde_tops = self._x_tildes.topk(self.top_m).indices  # [n_items, top_m]
        dummy_column = torch.arange(self.n_items)[:, None].to(self._x_tildes.device)
        m[dummy_column, x_tilde_tops] = True

        # Creating approximated mask
        m_tilde = nn.Softmax(dim=-1)(self._x_tildes / self.temp_masking)  # [n_items, n_clusters]

        # Putting together the masks
        m_hat = m_tilde + (m - m_tilde).detach()

        # Building affiliation vector
        self._xs = nn.Sigmoid()(self._x_tildes) * m_hat

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        a_i, _ = u_repr
        x_i, _ = i_repr

        sparse_dots = (a_i.unsqueeze(-2) * x_i).sum(dim=-1)
        return sparse_dots

    def get_item_representations_pre_tune(self, i_idxs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # i_idxs is ignored
        i_embed = self.item_embed.weight  # [n_items, embed_d]
        self._x_tildes = compute_cosine_sim(i_embed, self.clusters)  # [n_items, n_clusters]

        # Creating exact mask
        m = torch.zeros_like(self._x_tildes).to(self._x_tildes.device)
        x_tilde_tops = self._x_tildes.topk(self.top_m).indices  # [n_items, top_m]
        dummy_column = torch.arange(self.n_items)[:, None].to(self._x_tildes.device)
        m[dummy_column, x_tilde_tops] = True

        # Creating approximated mask
        m_tilde = nn.Softmax(dim=-1)(self._x_tildes / self.temp_masking)  # [n_items, n_clusters]

        # Putting together the masks
        m_hat = m_tilde + (m - m_tilde).detach()

        # Building affiliation vector
        self._xs = nn.Sigmoid()(self._x_tildes) * m_hat
        return self._xs, self.item_embed.weight

    def get_item_representations_post_tune(self, i_repr: torch.Tensor) -> Union[
        torch.Tensor, Tuple[torch.Tensor, ...]]:

        return i_repr

    def get_user_representations_pre_tune(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        y_u = self.interaction_matrix[u_idxs]  # [batch_size, n_items]
        u_embed = self.user_embed(u_idxs)
        a_tilde = y_u @ self._x_tildes  # [batch_size, n_clusters]

        # Creating exact mask
        m = torch.zeros_like(a_tilde).to(a_tilde.device)
        a_tilde_tops = a_tilde.topk(self.top_n).indices
        dummy_column = torch.arange(a_tilde.shape[0])[:, None].to(a_tilde.device)
        m[dummy_column, a_tilde_tops] = True

        # Creating approximated mask
        m_tilde = nn.Softmax(dim=-1)(a_tilde / self.temp_masking)

        # Putting together the masks
        m_hat = m_tilde + (m - m_tilde).detach()

        # Building affiliation vector
        a_i = nn.Sigmoid()(a_tilde) * m_hat

        return a_i, u_embed

    def get_user_representations_post_tune(self, u_repr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return u_repr

    def get_and_reset_other_loss(self) -> Dict:
        acc_ts, acc_ind, acc_cf = self._acc_ts, self._acc_ind, self._acc_cf
        self._acc_ts = self._acc_ind = self._acc_cf = 0
        cf_loss = self.lam_cf * acc_cf
        ind_loss = self.lam_ind * acc_ind
        ts_loss = self.lam_ts * acc_ts

        return {
            'reg_loss': ts_loss + ind_loss + cf_loss,
            'cf_loss': cf_loss,
            'ind_loss': ind_loss,
            'ts_loss': ts_loss
        }

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        init_signature = inspect.signature(ECF.__init__)
        def_parameters = {k: v.default for k, v in init_signature.parameters.items() if
                          v.default is not inspect.Parameter.empty}
        parameters = {**def_parameters, **conf}

        return ECF(dataset.n_users, dataset.n_items, dataset.tag_matrix,
                   dataset.sampling_matrix, parameters['embedding_dim'],
                   parameters['n_clusters'], parameters['top_n'], parameters['top_m'],
                   parameters['temp_masking'], parameters['temp_tags'],
                   parameters['top_p'], parameters['lam_cf'], parameters['lam_ind'],
                   parameters['lam_ts']
                   )

    def to(self, *args, **kwargs):
        for arg in args:
            if type(arg) == torch.device or arg == 'cuda' or arg == 'cpu':
                self.tag_matrix = self.tag_matrix.to(arg)
                self.interaction_matrix = self.interaction_matrix.to(arg)
        return super().to(*args, **kwargs)

    def load_model_from_path(self, path: str):
        path = os.path.join(path, 'model.pth')
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=False)
        print('Model Loaded')


class DeepMatrixFactorization(SGDBasedRecommenderAlgorithm):
    """
    Deep Matrix Factorization Models for Recommender Systems by Xue et al. (https://www.ijcai.org/Proceedings/2017/0447.pdf)
    """

    def __init__(self, dataset: InteractionRecDataset, u_mid_layers: Union[List[int], int],
                 i_mid_layers: Union[List[int], int], final_dimension: int, mu: float = 1.e-6,
                 normalize_interactions: bool = False, normalize_representations: bool = False,
                 use_output_activation_fn: bool = False):
        """
        :param dataset: the dataset containing all user-item interactions
        :param u_mid_layers: list of integers representing the size of the middle layers on the user side
        :param i_mid_layers: list of integers representing the size of the middle layers on the item side
        :param final_dimension: last dimension of the layers for both user and item side
        :param mu: parameter that sets the minimum for the prediction of DeepMF. See equation (13) of the DeepMF paper
        :param normalize_interactions: whether to normalize the interactions so that the number of interactions of
                                       a single user (item) do not influence the resulting user (item) representations
        :param normalize_representations: whether to l2-normalize the user (item) representations
        :param use_output_activation_fn: whether to use an activation function on the user & item output
        """
        super().__init__()
        self.dataset = dataset
        self.normalize_interactions = normalize_interactions
        self.normalize_representations = normalize_representations

        self.mu = mu

        self.final_dimension = final_dimension

        if isinstance(u_mid_layers, int):
            u_mid_layers = [u_mid_layers]
        if isinstance(i_mid_layers, int):
            i_mid_layers = [i_mid_layers]

        self.u_layers = [self.dataset.n_items] + u_mid_layers + [self.final_dimension]
        self.i_layers = [self.dataset.n_users] + i_mid_layers + [self.final_dimension]

        # create embedding layers, output activation as done in original paper
        output_fn = nn.ReLU() if use_output_activation_fn else None
        self.user_nn = PolyLinear(self.u_layers, activation_fn=nn.ReLU(), output_fn=output_fn)
        self.item_nn = PolyLinear(self.i_layers, activation_fn=nn.ReLU(), output_fn=output_fn)

        self.cosine_func = nn.CosineSimilarity(dim=-1)

        # Initialization of the network
        self.user_nn.apply(general_weight_init)
        self.item_nn.apply(general_weight_init)

        print(f'Built {self.name} module \n'
              f'- u_layers: {self.u_layers} \n'
              f'- i_layers: {self.i_layers} \n')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor):
        u_vec = self.get_user_representations(u_idxs)
        i_vec = self.get_item_representations(i_idxs)
        sim = self.combine_user_item_representations(u_vec, i_vec)
        return sim

    @staticmethod
    def build_from_conf(conf: dict, train_dataset: TrainRecDataset):
        return DeepMatrixFactorization(dataset=train_dataset, u_mid_layers=conf.get('u_mid_layers', []),
                                       i_mid_layers=conf.get('i_mid_layers', []),
                                       final_dimension=conf['final_dimension'], mu=conf.get('mu', 1e-6),
                                       normalize_interactions=conf.get('normalize_interactions', False),
                                       normalize_representations=conf.get('normalize_representations', False),
                                       use_output_activation_fn=conf.get('use_output_activation_fn', False))

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        with torch.no_grad():
            i_vec = self.dataset.get_item_interaction_vectors(i_idxs).float()
            if self.normalize_interactions:
                norm = torch.linalg.vector_norm(i_vec, dim=-1, keepdim=True)
                i_vec = i_vec / norm.clamp(min=1e-8)  # prevent division by 0

        i_vec = self.item_nn(i_vec)

        if self.normalize_representations:
            norm = torch.linalg.vector_norm(i_vec, dim=-1, keepdim=True)
            i_vec = i_vec / norm.clamp(min=1e-8)  # prevent division by 0

        return i_vec

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        with torch.no_grad():
            u_vec = self.dataset.get_user_interaction_vectors(u_idxs).float()
            if self.normalize_interactions:
                norm = torch.linalg.vector_norm(u_vec, dim=-1, keepdim=True)
                u_vec = u_vec / norm.clamp(min=1e-8)  # prevent division by 0

        u_vec = self.user_nn(u_vec)

        if self.normalize_representations:
            norm = torch.linalg.vector_norm(u_vec, dim=-1, keepdim=True)
            u_vec = u_vec / norm.clamp(min=1e-8)  # prevent division by 0

        return u_vec

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        sim = self.cosine_func(u_repr[:, None, :], i_repr)
        sim[sim < self.mu] = self.mu
        return sim

    def load_model_from_path(self, path: str):
        path = os.path.join(path, 'model.pth')
        state_dict = torch.load(path)

        # In previous DMF version, we also saved the user and item interactions in the
        # model file.
        if 'user_vectors.weight' in state_dict:
            # don't need those weights as we retrieve them directly from the dataset
            state_dict.pop('user_vectors.weight')
            state_dict.pop('item_vectors.weight')

            def to_new_key(k):
                """
                mapping function for previous parameter names to new parameter names
                example:
                    old keys: user_nn.0.weight, user_nn.2.weight
                    new keys: user_nn.layers.linear_0.weight, user_nn.layers.linear_1.weight
                """

                pattern = r'^(\w+)\.(\d+)\.(\w+)$'
                match_obj = re.match(pattern, k)

                if match_obj is not None:
                    old_layer_nr = int(match_obj[2])
                    if old_layer_nr % 2 != 0:
                        raise ValueError('did not expect an odd layer number for old-version parameter names')
                    return f'{match_obj[1]}.layers.linear_{int(old_layer_nr / 2)}.{match_obj[3]}'
                return k

            state_dict = {to_new_key(k): v for k, v in state_dict.items()}

        self.load_state_dict(state_dict)
        print('Model Loaded')


class FeatureEmbedding(nn.Module):
    def __init__(self, feature: Feature, embedding_dim: int = None,
                 pre_embedding_layers: list[int] = None, post_embedding_layers: list[int] = None,
                 activation_fn: str | nn.Module = 'relu'):
        """
        A module that allows for easy embedding and processing of any kind of feature.

        :param feature: The feature to embed
        :param embedding_dim: (optional)
            - for categorical and tag features: The size of the embedding module
            - for other feature types: Whether and if, how the features should be initially transformed
        :param pre_embedding_layers: (optional) Defines what kind of processing should be done on the raw input.
                                                This is not supported for tag & categorical features.
        :param post_embedding_layers: (optional) defines the size of linear layers that are applied after embedding
                                                 the feature. [100, 20] leads to two linear layers, which map from
                                                 feature_dim -> 100 -> 20
        :param activation_fn: (optional) the activation function for any transformations, applied after a linear layer
        """
        super().__init__()
        # store parameters
        self._feature = feature
        self._feature_type = feature.feature_definition.type
        self._embedding_dim = embedding_dim
        self._pre_embedding_layer_config = pre_embedding_layers
        self._post_embedding_layer_config = post_embedding_layers
        self._activation_fn = activation_fn

        # keeps track of final output dimension of feature
        self.output_dim = None

        # ensure that we have all parameters that we need
        if embedding_dim is None and self._feature_type in (FeatureType.CATEGORICAL, FeatureType.TAG):
            raise ValueError(f'For {self._feature_type} feature "{self._feature.feature_definition.name}", '
                             f'the size of its embeddings have to be specified with "embedding_dim"')

        # ensure validity of parameters
        if pre_embedding_layers and self._feature_type in (FeatureType.CATEGORICAL, FeatureType.TAG):
            raise ValueError(f'For {self._feature_type} feature "{self._feature.feature_definition.name}", '
                             f'using pre-embedding layers would not make any sense (as the input are simple indices).')

        # create layers that are / might be necessary for features
        self.pre_embedding_layers = self._create_pre_embedding_layers()
        self.embedding_layer = self._create_embedding_layer()
        self.post_embedding_layers = self._create_post_embedding_layers()

        # initialize the different weights
        self.apply(general_weight_init)

    def _create_embedding_layer(self):
        match self._feature_type:
            case FeatureType.CATEGORICAL:
                self.output_dim = self._embedding_dim
                return nn.Embedding(self._feature.n_unique_categories, self._embedding_dim)
            case FeatureType.TAG:
                # use embedding bag, which sums up the embeddings for the individual tags so that
                # we have just a single embedding moving forward
                self.output_dim = self._embedding_dim
                return nn.EmbeddingBag(self._feature.dim + 1, self._embedding_dim,  # +1 to account for padding
                                       padding_idx=-1)  # last embedding is pad embedding
            case _:
                # other features do not require an embedding
                return None

    def _create_pre_embedding_layers(self):
        if self._feature_type not in (FeatureType.CATEGORICAL, FeatureType.TAG):
            layer_config = [self._feature.dim]

            if self._pre_embedding_layer_config:
                # use pre-embedding layers to transform features
                layer_config.extend(self._pre_embedding_layer_config)

            if self._embedding_dim is not None:
                # project feature to the embedding dimension
                layer_config.append(self._embedding_dim)

            self.output_dim = layer_config[-1]
            if len(layer_config) > 1:
                return PolyLinear(layer_config, activation_fn=self._activation_fn, output_fn=self._activation_fn)
        return None

    def _create_post_embedding_layers(self):
        layer_config = [self.output_dim]

        if self._post_embedding_layer_config:
            # encoding layers follow after the initial embedding layers
            layer_config.extend(self._post_embedding_layer_config)

        if len(layer_config) > 1:
            # create encoding layers
            self.output_dim = layer_config[-1]
            return PolyLinear(layer_config, activation_fn=self._activation_fn, output_fn=self._activation_fn)

        return None

    def forward(self, indices):
        # retrieve features
        features = self._feature[indices]
        x = features.to(device=indices.device)

        # process features
        if self.pre_embedding_layers is not None:
            x = self.pre_embedding_layers(x.float())
        if self.embedding_layer is not None:
            # nn.Embedding and nn.EmbeddingBag work differently on different input shapes
            if isinstance(self.embedding_layer, nn.EmbeddingBag) and x.dim() == 3:
                x = torch.vmap(self.embedding_layer)(x)
            else:
                x = self.embedding_layer(x.long())
        if self.post_embedding_layers is not None:
            x = self.post_embedding_layers(x.float())
        return x

    @classmethod
    def build_from_conf(cls, config: FeatureModuleConfig, feature: Feature):
        conf = config.to_dict()
        # feature name is not used, as it is already supplied by 'feature'
        conf.pop('feature_name')
        return cls(feature, **conf)


class ItemFeatureMatrixFactorization(SGDMatrixFactorization):
    """
    Implements a hybrid matrix factorization algorithm that uses a contrastive
    loss to push item embeddings and feature representations to be close.
    """

    def __init__(self, dataset: RecDataset, feature_name: str,
                 aggregate_for_rec: bool = False, lambda_content: float = 0.0001, temperature: float = 0.1,
                 embedding_loss_aggregator: str = 'mean', intermediate_layers=None,
                 embedding_dim: int = 100, use_user_bias: bool = False,
                 use_item_bias: bool = False, use_global_bias: bool = False):
        """
        :param dataset: The dataset used to retrieve the content features of
        :param feature_name: The item feature to use for item based feature matrix factorization
        :param aggregate_for_rec: Whether the item representation to use for recommendation should be an average of the
                                  MF embedding
        and of the item content embedding, or not. Default is False, i.e., only use the item embedding
        :param lambda_content: The relative weight of the contrastive loss
        :param temperature: The temperature to use in rescaling the logits of the contrastive loss
        :param embedding_loss_aggregator: The aggregating function ['mean', 'sum'] to use when computing the
                                          contrastive loss
        :param intermediate_layers: The intermediate layers to be used to project the features to the same
                                    dimensionality of the MF embedding
        """
        super().__init__(dataset.n_users, dataset.n_items, embedding_dim, use_user_bias, use_item_bias, use_global_bias)

        self.dataset = dataset
        self.feature_name = feature_name

        self.aggregate_for_rec = aggregate_for_rec
        self.lambda_content = lambda_content

        self.embedding_net = FeatureEmbedding(feature=dataset.item_features[feature_name],
                                              pre_embedding_layers=intermediate_layers,
                                              embedding_dim=embedding_dim)
        self.emb_loss_fn = InfoNCE(temperature, embedding_loss_aggregator)
        self.emb_loss = 0.

        logging.info(f'- feature: {self.feature_name} \n'
                     f'- aggregate_for_rec: {self.aggregate_for_rec} \n'
                     f'- lambda_content: {self.lambda_content} \n'
                     f'- emb_loss_fn: {self.emb_loss_fn}')

    def forward(self, u_idxs, i_idxs):
        """
        First we compute all item and user representations
        we then pass all of them to the combine method, which implicitly unpacks them
        we then explicitly unpack them to pass them to the regularization loss
        """
        u_repr = self.get_user_representations(u_idxs)
        i_repr = self.get_item_representations(i_idxs)

        dots = self.combine_user_item_representations(u_repr, i_repr)
        item_profile_representations, item_content_representations, *_ = i_repr

        # contrastive loss
        self.compute_reg_losses(item_profile_representations, item_content_representations)
        return dots

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        This should return both profile and content representations
        and (potentially) item bias as well
        """
        item_profile_representations = self.item_embeddings(i_idxs)
        # Shape is (batch_size, n_neg + 1, feature_dim)
        item_content_representations = self.embedding_net(i_idxs)

        if self.use_item_bias:
            return item_profile_representations, item_content_representations, self.item_bias(i_idxs).squeeze()
        return item_profile_representations, item_content_representations

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """
        This should unpack the profile and content representations
        and choose which one to combine for the recommendation score.
        Then it calls the combine_user_item_representations of the
        SGDMatrixFactorization parent method
        """
        if self.aggregate_for_rec:
            i_embed = torch.stack(i_repr[:2], dim=0).mean(dim=0)
        else:
            i_embed = i_repr[0]

        if self.use_item_bias:
            return super().combine_user_item_representations(u_repr, (i_embed, i_repr[-1]))
        else:
            return super().combine_user_item_representations(u_repr, i_embed)

    def compute_reg_losses(self, profile_embs, content_embs):
        self.emb_loss = self.emb_loss_fn(profile_embs, content_embs)

    def get_and_reset_other_loss(self) -> Dict:
        emb_loss = self.emb_loss
        self.emb_loss = 0
        return {
            'reg_loss': emb_loss
        }

    @staticmethod
    def build_from_conf(conf: dict, dataset: RecDataset):
        return ItemFeatureMatrixFactorization(dataset, conf['feature_name'],
                                              conf['aggregate_for_rec'], conf['lambda_content'], conf['temperature'],
                                              conf['embedding_loss_aggregator'], conf['intermediate_layers'],
                                              conf['embedding_dim'], conf['use_user_bias'], conf['use_item_bias'],
                                              conf['use_global_bias'])


class UserFeatureMatrixFactorization(SGDMatrixFactorization):
    """
    Implements a hybrid matrix factorization algorithm that uses a contrastive
    loss to push user embeddings and feature representations to be close.
    """

    def __init__(self, dataset: RecDataset, feature_name: str,
                 aggregate_for_rec: bool = False, lambda_content: float = 0.0001, temperature: float = 0.1,
                 embedding_loss_aggregator: str = 'mean', intermediate_layers=None,
                 embedding_dim: int = 100, use_user_bias: bool = False,
                 use_item_bias: bool = False, use_global_bias: bool = False):
        """
        :param dataset: The dataset used to retrieve the content features of
        :param feature_name: The user feature to use for user based feature matrix factorization
        :param aggregate_for_rec: Whether the user representation to use for recommendation should be an average of the
                                  MF embedding and of the user content embedding, or not.
                                  Default is False, i.e., only use the user embedding
        :param lambda_content: The relative weight of the contrastive loss
        :param temperature: The temperature to use in rescaling the logits of the contrastive loss
        :param embedding_loss_aggregator: The aggregating function ['mean', 'sum'] to use when computing the
                                          contrastive loss
        :param intermediate_layers: The intermediate layers to be used to project the features to the same
                                    dimensionality of the MF embedding
        """
        super().__init__(dataset.n_users, dataset.n_items, embedding_dim, use_user_bias, use_item_bias, use_global_bias)

        self.dataset = dataset
        self.feature_name = feature_name

        self.aggregate_for_rec = aggregate_for_rec
        self.lambda_content = lambda_content

        self.embedding_net = FeatureEmbedding(feature=dataset.user_features[feature_name],
                                              pre_embedding_layers=intermediate_layers,
                                              embedding_dim=embedding_dim)
        self.emb_loss_fn = InfoNCE(temperature, embedding_loss_aggregator)
        self.emb_loss = 0.

        logging.info(f'- feature: {self.feature_name} \n'
                     f'- aggregate_for_rec: {self.aggregate_for_rec} \n'
                     f'- lambda_content: {self.lambda_content} \n'
                     f'- emb_loss_fn: {self.emb_loss_fn}')

    def forward(self, u_idxs, i_idxs):
        """
        First we compute all item and user representations
        we then pass all of them to the combine method, which implicitly unpacks them
        we then explicitly unpack them to pass them to the regularization loss
        """
        u_repr = self.get_user_representations(u_idxs)
        i_repr = self.get_item_representations(i_idxs)

        dots = self.combine_user_item_representations(u_repr, i_repr)
        user_profile_representations, user_content_representations, *_ = u_repr

        # contrastive loss (blow up first dimension to support InfoNCE loss)
        self.compute_reg_losses(user_profile_representations[:, None, :], user_content_representations[:, None, :])
        return dots

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        This should return both profile and content representations
        and (potentially) user bias as well
        """
        user_profile_representations = self.user_embeddings(u_idxs)
        # Shape is (batch_size, 1, feature_dim) ==> transform to (batch_size, feature_dim)
        user_content_representations = self.embedding_net(u_idxs).squeeze(dim=1)

        if self.use_user_bias:
            return user_profile_representations, user_content_representations, self.user_bias(u_idxs).squeeze()
        return user_profile_representations, user_content_representations

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """
        This should unpack the profile and content representations
        and choose which one to combine for the recommendation score.
        Then it calls the combine_user_item_representations of the
        SGDMatrixFactorization parent method
        """
        if self.aggregate_for_rec:
            u_embed = torch.stack(u_repr[:2], dim=0).mean(dim=0)
        else:
            u_embed = u_repr[0]

        if self.use_user_bias:
            return super().combine_user_item_representations((u_embed, u_repr[-1]), i_repr)
        else:
            return super().combine_user_item_representations(u_embed, i_repr)

    def compute_reg_losses(self, profile_embs, content_embs):
        self.emb_loss = self.emb_loss_fn(profile_embs, content_embs)

    def get_and_reset_other_loss(self) -> Dict:
        emb_loss = self.emb_loss
        self.emb_loss = 0
        return {
            'reg_loss': emb_loss
        }

    @staticmethod
    def build_from_conf(conf: dict, dataset: RecDataset):
        return UserFeatureMatrixFactorization(dataset, conf['feature_name'],
                                              conf['aggregate_for_rec'], conf['lambda_content'], conf['temperature'],
                                              conf['embedding_loss_aggregator'], conf['intermediate_layers'],
                                              conf['embedding_dim'], conf['use_user_bias'], conf['use_item_bias'],
                                              conf['use_global_bias'])


class DropoutNetEntity(nn.Module):
    def __init__(self, entity_name: str, preference_dim: int, features: Dict[str, Feature],
                 entity_config: DropoutNetEntityConfig, shared_common_dim: int):
        super().__init__()
        self.entity_name = entity_name
        self.features = features
        self.entity_config = entity_config
        self.shared_common_dim = shared_common_dim

        # create subnets for the different stages
        # ... for preference embeddings
        self.pref_net = PolyLinear([preference_dim] + entity_config.preference_layers)
        self.pref_dim = entity_config.preference_layers[-1]

        self.cont_dim = 0
        self.cont_modules = nn.ModuleList()
        for f in entity_config.features:
            feature_module = FeatureEmbedding.build_from_conf(f, features[f.feature_name])
            self.cont_modules.append(feature_module)
            self.cont_dim += feature_module.output_dim

        # ... to process concatenation of preference and content embeddings
        self._net_shape = (
                [self.pref_dim + self.cont_dim]
                + entity_config.common_hidden_layers
                + [shared_common_dim])
        self.net = PolyLinear(self._net_shape, activation_fn=entity_config.activation_fn)

    def forward(self, indices, preferences):
        # transform them with linear layer
        # For missing preferences, this will still add the bias term.
        # However, that should be fine --> are just extra parameters
        pref = self.pref_net(preferences)
        cont = [m(indices) for m in self.cont_modules]

        # as in original paper, concatenate features (note that preference vector may be a zero vector)
        x = torch.cat([*cont, pref], dim=-1)
        return self.net(x)


class DropoutNet(SGDBasedRecommenderAlgorithm):
    """
    DropoutNet: Addressing Cold Start in Recommender Systems
    Volkovs, Maksims and Yu, Guangwei and Poutanen, Tomi
    https://papers.nips.cc/paper_files/paper/2017/hash/dbd22ba3bd0df8f385bdac3e9f8be207-Abstract.html
    """

    def __init__(self, config: DropoutNetConfig, dataset: InteractionRecDataset):
        super().__init__()

        self.config = config
        self.dataset = dataset

        self.user_net = DropoutNetEntity('user', preference_dim=self.dataset.n_items,
                                         features=dataset.user_features, entity_config=config.user,
                                         shared_common_dim=config.shared_common_dim)
        self.item_net = DropoutNetEntity('item', preference_dim=self.dataset.n_users,
                                         features=dataset.item_features, entity_config=config.item,
                                         shared_common_dim=config.shared_common_dim)

        self._rng = np.random.default_rng(config.sampling_seed)
        logging.info(f'Built {self.name} module with config\n{config.to_dict()}')

    def sample_training_strategy(self, n_samples):
        if self.training:
            return self._rng.choice(DropoutNetSamplingStrategy.list(), size=n_samples, replace=True)
        else:
            # in validation, we want to use all available information
            return np.full(n_samples, fill_value=DropoutNetSamplingStrategy.Normal.value)

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Returns a user representation given the user indexes. It is especially useful for faster validation.
        :param u_idxs: user indexes. Shape is (batch_size)
        :return: user representation/s. The output depends on the model.
        """
        strategy = self.sample_training_strategy(len(u_idxs))
        indices = np.argwhere(strategy == DropoutNetSamplingStrategy.Normal.value).flatten()
        users_with_preferences = u_idxs[indices]

        # where we don't have interactions, we simply forward a zero-vector
        preferences = torch.zeros(size=(len(u_idxs), self.dataset.n_items),
                                  dtype=torch.float, device=u_idxs.device)

        if len(users_with_preferences) > 0:
            # retrieve preferences of users
            user_interactions = self.dataset.get_user_interaction_vectors(users_with_preferences)
            preferences[indices] = user_interactions.float()

        return self.user_net(u_idxs, preferences)

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Returns an item representation given the item indexes. It is especially useful for faster validation.
        :param i_idxs: item indexes. Shape is (batch_size)
        :return: item representation/s. The output depends on the model.
        """
        strategy = self.sample_training_strategy(len(i_idxs))
        indices = np.argwhere(strategy == DropoutNetSamplingStrategy.Normal.value).flatten()
        items_with_preferences = i_idxs[indices]

        # where we don't have interactions, we simply forward a zero-vector
        preferences = torch.zeros(size=(i_idxs.shape + (self.dataset.n_users,)),
                                  dtype=torch.float, device=i_idxs.device)

        if len(items_with_preferences) > 0:
            # retrieve actual preferences of items
            item_interactions = self.dataset.get_item_interaction_vectors(items_with_preferences)
            preferences[indices] = item_interactions.float()

        return self.item_net(i_idxs, preferences)

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """
        Combine the user and item representations to generate the final logits.
        :param u_repr: User representations (see get_user_representations)
        :param i_repr: Item representations (see get_item_representations)
        :return:
        """
        # happens if we get item embeddings, not just 1 + n_neg
        if i_repr.ndim == 2:
            # perform dot product on last dimension
            # (b ... batch_size, e ... embedding_dim, c ... n_items)
            return torch.einsum('be, ce -> bc', u_repr, i_repr)

        # perform dot product on last dimension (b ... batch_size, s ... 1, c ... 1+n_neg, e ... embedding_dim)
        # u_repr ... [batch_size, embedding_dim]
        # i_repr ... [batch_size, 1+n_neg, embedding_dim]
        return torch.einsum('be, bce -> bc', u_repr, i_repr)

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        """
        Similar to predict but used for training. It provides a simple default implementation that can be adjusted in
        case.
        """
        u_repr = self.get_user_representations(u_idxs)
        i_repr = self.get_item_representations(i_idxs)

        out = self.combine_user_item_representations(u_repr, i_repr)
        return out

    @staticmethod
    def build_from_conf(conf: dict, dataset: InteractionRecDataset):
        return DropoutNet(DropoutNetConfig.from_dict(conf), dataset)


class SingleBranchNetEntity(nn.Module):
    def __init__(self, entity_name: str, features: Dict[str, Feature],
                 entity_config: SingleBranchNetEntityConfig, shared_common_dim: int,
                 val_interactions_available: bool = True):
        super().__init__()
        self.features = features
        self.entity_name = entity_name
        self.entity_config = entity_config
        self.output_dim = shared_common_dim
        self.val_interactions_available = val_interactions_available

        if len(entity_config.features) == 0:
            raise ValueError('SingleBranchEntity requires at least one feature.')

        self.train_modalities = self._get_modalities(train=True)
        self.eval_modalities = self._get_modalities(train=False)

        self._available_feature_modalities = set(features.keys())
        not_available_modalities = self.train_modalities - self._available_feature_modalities
        if len(not_available_modalities) > 0:
            raise ValueError(f'Features for modalities {not_available_modalities} are not available!')

        self._available_feature_definitions = set(f.feature_name for f in entity_config.features)
        not_available_definitions = self.train_modalities - self._available_feature_definitions
        if len(not_available_definitions) > 0:
            raise ValueError(f'Network definitions for modalities {not_available_definitions} are not available!')

        # create subnets for the different modalities
        self.modality_modules = nn.ModuleDict()
        # ... for the different features of the entity
        for f in entity_config.features:
            # ignore feature if we do not use it during training
            if f.feature_name not in self.train_modalities:
                continue

            # collect parameters for feature module
            feature_conf = FeatureModuleConfig(
                feature_name=f.feature_name,
                embedding_dim=entity_config.common_modality_dim,
                pre_embedding_layers=f.feature_hidden_layers,
                activation_fn=entity_config.activation_fn
            )

            feature_module = FeatureEmbedding.build_from_conf(feature_conf, features[f.feature_name])
            self.modality_modules[f.feature_name] = feature_module

        print(f'sb net {self.train_modalities}, modules for {list(self.modality_modules.keys())}')

        # define single branch network that encodes all modalities with the same network
        sb_net_layers = []
        if self.entity_config.single_branch_input_dropout is not None:
            sb_net_layers.append(nn.Dropout(self.entity_config.single_branch_input_dropout))

        # determine how to do batch normalization like this to be backward compatible with simple batch-normalization
        # after the linear layers
        apply_batch_norm_every = entity_config.apply_batch_norm_every if entity_config.apply_batch_normalization else 0

        sb_net_layers.append(
            PolyLinear(
                [entity_config.common_modality_dim]
                + entity_config.single_branch_hidden_layers
                + [self.output_dim],
                activation_fn=entity_config.activation_fn,
                output_fn=entity_config.activation_fn if entity_config.apply_output_activation else None,
                apply_batch_norm_every=apply_batch_norm_every
            )
        )

        # still here for backward compatibility to support loading experiments of previous code versions,
        # can be dropped at one point in time
        if entity_config.apply_batch_normalization and entity_config.apply_batch_norm_every == 0:
            # original paper applies batch normalization as last layer
            # (actually also inbetween, but we'll ignore that for now)
            sb_net_layers.append(nn.BatchNorm1d(self.output_dim))

        self.sb_net = nn.Sequential(*sb_net_layers)
        if entity_config.aggregation_fn not in AGGREGATION_FUNCTIONS:
            raise ValueError(f'Aggregation function "{entity_config.aggregation_fn}" is not supported.')
        self.aggregation_fn = AGGREGATION_FUNCTIONS[entity_config.aggregation_fn]

        self.regularization_loss_fn = ZeroLossModule()
        if entity_config.embedding_regularization_type != EmbeddingRegularizationType.NoRegularization:
            self.regularization_loss_fn = InfoNCE(temperature=self.entity_config.regularization_temperature)
        self.regularization_loss = self._zero_loss()
        self._rng = np.random.default_rng(entity_config.sampling_seed)

    def forward(self, indices):
        # either shape (batch_size, 1 + n_neg, n_embeddings) or (batch_size, n_embeddings)
        modalities = self._sample_modalities(indices)

        # either shape (batch_size, 1 + n_neg, n_embeddings, emb_dim) or (batch_size, n_embeddings, emb_dim)
        single_branch_encoded_embeddings = self._embed(indices, modalities)
        if self.training:
            self.compute_reg_losses(single_branch_encoded_embeddings)

        # aggregate all retrieved embeddings per indices
        # leads to shape (batch_size, 1 + n_neg, embedding_dim) or (batch_size, embedding_dim)
        modality_embeddings = self.aggregation_fn(single_branch_encoded_embeddings, dim=-2)

        return modality_embeddings

    def _embed(self, indices, modalities):
        # either shape (batch_size, 1 + n_neg, n_embeddings, embedding_dim) or (batch_size, n_embeddings, embedding_dim)
        modality_embeddings = self._get_modality_embeddings(indices, modalities)
        modality_shape = modality_embeddings.shape

        # push through single branch network
        modality_embeddings = modality_embeddings.view(-1, self.entity_config.common_modality_dim)

        if self.entity_config.normalize_single_branch_input:
            modality_embeddings = nn.functional.normalize(modality_embeddings, p=2, dim=-1)

        single_branch_encoded_embeddings = self.sb_net(modality_embeddings)
        return single_branch_encoded_embeddings.view(*modality_shape[:-1], -1)

    def _get_modalities(self, train=True):
        available_mods = {f.feature_name for f in self.entity_config.features}
        if train:
            mods = set(self.entity_config.train_modalities or available_mods)
        else:
            train_mods = self._get_modalities(train=True)

            # ensure validity of modalities
            if self.entity_config.eval_modalities is not None:
                for m in self.entity_config.eval_modalities:
                    if m not in train_mods:
                        raise ValueError(f'Cannot use modality "{m}" during evaluation, '
                                         f'if it is not used during training.')

            # for evaluation, either use configured modalities, or train modalities
            mods = set(self.entity_config.eval_modalities or train_mods)
            # cannot use interaction embeddings if they are not available during evaluation
            if not self.val_interactions_available:
                mods.discard('interactions')

        if len(mods) == 0:
            raise ValueError(f'No single modality is available during {"training" if train else "evaluation"}: '
                             f'There are either no modalities specified or no interactions are available)')
        return mods

    def _sample_modalities(self, indices):
        if self.training:
            common_sampling_params = {
                'a': list(self.train_modalities),
                'size': indices.shape,
                'rng': self._rng
            }

            match self.entity_config.embedding_regularization_type:
                case EmbeddingRegularizationType.NoRegularization:
                    # sample just a single modality per indices
                    return row_wise_sample(**common_sampling_params, k=1)

                case EmbeddingRegularizationType.PairwiseSingle:
                    # sample two available modalities per indices
                    return row_wise_sample(**common_sampling_params, k=2)

                case EmbeddingRegularizationType.CentralModality:
                    # sample 1 other modality in addition to the central modality
                    return row_wise_sample(**common_sampling_params, k=2,
                                           central_item=self.entity_config.central_modality)
                case _:
                    raise ValueError(f'Embedding regularization "{self.entity_config.embedding_regularization_type}"'
                                     f'is not yet supported.')
        else:
            # select all modalities during evaluation (a weighting of them, e.g., their average, should be taken
            mods = np.empty(shape=indices.shape + (len(self.eval_modalities),), dtype=object)
            mods[None:] = list(self.eval_modalities)
            return mods

    def _get_modality_embeddings(self, indices: torch.Tensor, modalities: np.ndarray):
        """
        Samples and transform the different modalities for the specified indices
        """
        if indices.shape != modalities.shape[:-1]:
            raise ValueError('Shape of indices and modalities (up to the last dimension) does not match.')

        n_modalities_per_sample = modalities.shape[-1]

        # it's easier to process a 1d tensor
        indices_shape = indices.shape
        indices = indices.reshape(-1)
        indices = torch.repeat_interleave(indices, n_modalities_per_sample)
        modalities = modalities.reshape(-1)

        # check which modalities are used (to retrieve them semi-batch-wise)
        unique_modalities = np.unique(modalities)

        # create mask to see which modalities are used by which sample
        mask = modalities[:, None] == unique_modalities

        # create empty vector to store all embeddings
        all_embeddings = torch.empty(size=(indices.numel(), self.entity_config.common_modality_dim),
                                     device=indices.device)

        # retrieve all embeddings of a given modality and store it
        for i, mod in enumerate(unique_modalities.tolist()):
            # check which rows match the modality
            mod_ind = np.argwhere(mask[:, i]).flatten()

            # determine the sample indices, for which we have to retrieve the specific modality
            indices_with_modality = indices[mod_ind]

            # fetch modality
            embeddings_for_modality = self.modality_modules[mod](indices_with_modality)

            # reshape to make sure that sizes match
            embeddings = embeddings_for_modality.reshape(len(mod_ind), self.entity_config.common_modality_dim)

            # store modality
            all_embeddings[mod_ind] = embeddings

        all_embeddings = all_embeddings.reshape(indices_shape +
                                                (n_modalities_per_sample, self.entity_config.common_modality_dim))
        return all_embeddings

    def compute_reg_losses(self, embeddings: torch.Tensor):
        match self.entity_config.embedding_regularization_type:
            case EmbeddingRegularizationType.NoRegularization:
                self.regularization_loss = self._zero_loss()

            case EmbeddingRegularizationType.PairwiseSingle | EmbeddingRegularizationType.CentralModality:
                # ensure that we catch possible bugs
                if embeddings.shape[-2] != 2:
                    raise SystemError('second last dimension of embeddings should be of size 2')
                self.regularization_loss = self.regularization_loss_fn(embeddings[..., 0, :], embeddings[..., 1, :])

            case _:
                raise ValueError(f'Embedding regularization "{self.entity_config.embedding_regularization_type}"'
                                 f'is not yet supported.')

    def _zero_loss(self):
        return torch.tensor([0.], device=next(iter(self.parameters())).device)

    def _reset_reg_losses(self):
        self.regularization_loss = self._zero_loss()

    def get_and_reset_other_loss(self) -> Dict:
        loss = self.regularization_loss * self.entity_config.regularization_weight
        self._reset_reg_losses()
        return {
            'reg_loss': loss
        }


class SingleBranchNet(SGDBasedRecommenderAlgorithm):
    """
    M. S. Saeed et al., "Single-branch Network for Multimodal Training,"
    ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
    Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10097207.
    https://ieeexplore.ieee.org/abstract/document/10097207
    """

    def __init__(self, config: SingleBranchNetConfig, dataset: TrainRecDataset):
        super().__init__()

        # create user entity net
        user_features = dataset.user_features
        # ... user interactions are always a key part
        user_features['interactions'] = Feature(
            feature_definition=FeatureDefinition('interactions', FeatureType.VECTOR),
            raw_values=dataset.user_sampling_matrix_train
        )

        # treat user id as categorical feature so that we can easily get embeddings for them
        user_features['user_embedding'] = Feature(
            feature_definition=FeatureDefinition('user_embedding', FeatureType.CATEGORICAL),
            raw_values=np.arange(dataset.n_users)
        )

        self.is_user_sb_module = config.is_user_sb_module
        if self.is_user_sb_module:
            self.user_embedding_module = SingleBranchNetEntity(entity_name='user',
                                                               entity_config=config.user,
                                                               shared_common_dim=config.shared_common_dim,
                                                               features=user_features,
                                                               val_interactions_available=not dataset.is_cold_start_user)
        else:
            user_conf = config.user
            if user_conf.embedding_dim == -1:
                user_conf.embedding_dim = config.shared_common_dim
            self.user_embedding_module = FeatureEmbedding.build_from_conf(config.user,
                                                                          user_features[config.user.feature_name])

        # create item entity net
        item_features = dataset.item_features
        # ... item interactions are always a key part
        item_features['interactions'] = Feature(
            feature_definition=FeatureDefinition('interactions', FeatureType.VECTOR),
            raw_values=dataset.item_sampling_matrix_train
        )

        item_features['item_embedding'] = Feature(
            feature_definition=FeatureDefinition('item_embedding', FeatureType.CATEGORICAL),
            raw_values=np.arange(dataset.n_items)
        )

        self.is_item_sb_module = config.is_item_sb_module
        if self.is_item_sb_module:
            self.item_embedding_module = SingleBranchNetEntity(entity_name='item',
                                                               entity_config=config.item,
                                                               shared_common_dim=config.shared_common_dim,
                                                               features=item_features,
                                                               val_interactions_available=not dataset.is_cold_start_item)
        else:
            item_conf = config.item
            if item_conf.embedding_dim == -1:
                item_conf.embedding_dim = config.shared_common_dim
            self.item_embedding_module = FeatureEmbedding.build_from_conf(config.item,
                                                                          item_features[config.item.feature_name])

        logging.info(f'Built {self.name} module')

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Returns a user representation given the user indexes. It is especially useful for faster validation.
        :param u_idxs: user indexes. Shape is (batch_size)
        :return: user representation/s. The output depends on the model.
        """
        return self.user_embedding_module(u_idxs)

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Returns an item representation given the user indexes. It is especially useful for faster validation.
        :param i_idxs: item indexes. Shape is (batch_size, n_neg + 1), where n_neg is the number of negative samples
        :return: item representation/s. The output depends on the model.
        """
        return self.item_embedding_module(i_idxs)

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """
        Combine the user and item representations to generate the final logits.
        :param u_repr: User representations (see get_user_representations)
        :param i_repr: Item representations (see get_item_representations)
        :return:
        """
        # simplify matrix multiplication by using einsum
        # https://pytorch.org/docs/stable/generated/torch.einsum.html
        # this is basically a dot product between the last dimensions of both arrays

        # happens if we get item embeddings, not just 1 + n_neg
        if i_repr.ndim == 2:
            # perform dot product on last dimension
            # (b ... batch_size, e ... embedding_dim, c ... n_items)
            return torch.einsum('be, ce -> bc', u_repr, i_repr)

        # perform dot product on last dimension (b ... batch_size, s ... 1, c ... 1+n_neg, e ... embedding_dim)
        # u_repr ... [batch_size, embedding_dim]
        # i_repr ... [batch_size, 1+n_neg, embedding_dim]
        return torch.einsum('be, bce -> bc', u_repr, i_repr)

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        """
        Similar to predict but used for training. It provides a simple default implementation that can be adjusted in
        case.
        """
        u_repr = self.get_user_representations(u_idxs)
        i_repr = self.get_item_representations(i_idxs)

        out = self.combine_user_item_representations(u_repr, i_repr)
        return out

    def get_and_reset_other_loss(self) -> Dict:
        losses = {'reg_loss': torch.tensor([0.]).to(self.device)}

        if self.is_user_sb_module:
            user_reg_losses = self.user_embedding_module.get_and_reset_other_loss()
            losses['reg_loss'] += user_reg_losses['reg_loss']
            losses.update({f'user_{k}': v for k, v in user_reg_losses.items()})

        if self.is_item_sb_module:
            item_reg_losses = self.item_embedding_module.get_and_reset_other_loss()
            losses['reg_loss'] += item_reg_losses['reg_loss']
            losses.update({f'item_{k}': v for k, v in item_reg_losses.items()})

        return losses

    @staticmethod
    def build_from_conf(conf: dict, dataset: TrainRecDataset):
        return SingleBranchNet(SingleBranchNetConfig.from_dict(conf), dataset)
