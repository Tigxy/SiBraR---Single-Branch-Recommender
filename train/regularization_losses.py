import math

import torch
from torch import nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    def __init__(self, temperature: float = 1., loss_aggregator: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.loss_aggregator = loss_aggregator

    def forward(self, first_emb, second_emb) -> torch.Tensor:
        """
        Compute the symmetric cross entropy loss (InfoNCE)
        between item profile and item content embeddings.

        Similar to CLIP
        https://openai.com/research/clip

        @param first_emb: Tensor containing the first embeddings.
                          Shape is (batch_size, d1, ..., dk, 1 + n_negs, latent_dim)
        @param second_emb: Tensor containing the second embeddings.
                           Shape is (batch_size, d1, ..., dk, 1 + n_negs, latent_dim)
        """
        # [batch_size, d1, ..., dk, 1 + n_negs, 1 + n_negs]
        logits = first_emb @ second_emb.transpose(-2, -1) / self.temperature

        # Positive keys are the entries on the diagonal, therefore these are the correct labels:
        # [batch_size, d1, ..., dk, 1 + n_negs]
        labels = torch.arange(logits.shape[-1], device=first_emb.device).repeat(1, *logits.shape[:-2], 1)

        # Logits change depending on which modality we are "retrieving"
        logits_c_to_p = logits.reshape(-1, logits.shape[-1])
        logits_p_to_c = logits.transpose(-2, -1).reshape(-1, logits.shape[-1])

        labels = labels.reshape(-1)
        x_y_loss = F.cross_entropy(logits_c_to_p, labels, reduction=self.loss_aggregator)
        y_x_loss = F.cross_entropy(logits_p_to_c, labels, reduction=self.loss_aggregator)

        total_loss = x_y_loss + y_x_loss
        return total_loss


class ZeroLossModule(nn.Module):
    """
    Dummy module to return 0
    """
    def forward(self, *args, **kwargs):
        return torch.tensor([0])
