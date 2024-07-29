from typing import Callable

import numpy as np
from torch.autograd.profiler import record_function


def neg_samp_vectorized_bsearch(pos_indices, n_items, size=32):
    """
    Pre-verified with binary search `pos_indices` is assumed to be ordered
    https://medium.com/@2j/negative-sampling-in-numpy-18a9ad810385
    """
    raw_samples = np.random.choice(n_items - len(pos_indices), size=size, replace=False)
    pos_indices_adj = pos_indices - np.arange(len(pos_indices))
    ss = np.searchsorted(pos_indices_adj, raw_samples, side='right')
    neg_indices = raw_samples + ss
    return neg_indices


def negative_sample_uniform(choices: np.ndarray, size: int, positive_indices: np.ndarray = None):
    with record_function("validity_check"):
        if len(choices) - len(positive_indices) < size:
            # is unlikely to happen in our recommender task, and a minor overhead,
            # but still worth it for the added clarity if it should ever occur
            raise ValueError(f'Not enough values in the range to sample "{size}" unique values.')

    with record_function("map_indices"):
        pos_samp = np.searchsorted(choices, positive_indices)
    with record_function("sample_negative"):
        neg_samp = neg_samp_vectorized_bsearch(pos_samp, len(choices), size)
    with record_function("retrieve_indices"):
        negative_indices = choices[neg_samp]
    return negative_indices


def negative_sample_uniform_recbole(choices: np.ndarray, size: int, positive_indices: np.ndarray = None):
    n_choices = len(choices)
    with record_function("validity_check"):
        n_positive = len(positive_indices)
        if n_choices - n_positive < size:
            # is unlikely to happen in our recommender task, and a minor overhead,
            # but still worth it for the added clarity if it should ever occur
            raise ValueError(f'Not enough values in the range to sample "{size}" unique values.')

        if (n_choices - n_positive) * 0.5 < size:
            raise ValueError(f'Sampling is really inefficient either because the number of choices are small'
                             f'or the number of items to sample is too high.')

    with record_function("setup_arrays"):
        neg_samples = np.full(size, fill_value=-1)
        neg_ids_to_still_sample = list(range(size))

    with record_function("do_sampling"):
        # sample as long as we have not sampled enough items
        while len(neg_ids_to_still_sample):
            # just sample from the whole collection of choices
            neg_samples[neg_ids_to_still_sample] = np.random.randint(low=0, high=n_choices,
                                                                     size=len(neg_ids_to_still_sample))

            # determine indices that we have to resample
            neg_ids_to_still_sample = [i for i, v in zip(neg_ids_to_still_sample, neg_samples[neg_ids_to_still_sample])
                                       if v in positive_indices]

    with record_function("retrieve_indices"):
        neg_samples = choices[neg_samples]

    return neg_samples


def negative_sample_popular(choices: np.ndarray, size: int, popularity_distribution: np.ndarray,
                            squashing_factor: float, positive_indices: np.ndarray = None):
    if positive_indices is not None:
        choices = np.setdiff1d(choices, positive_indices, assume_unique=True)

    # select only popularity for the items we actually care for
    p = popularity_distribution[choices]
    p = np.power(p, squashing_factor)  # Applying squashing factor alpha

    # normalize popularity to sum up to 1.
    p = p / p.sum()
    return np.random.choice(choices, size=size, p=p)


def sample_multiple(fn: Callable, n_samples: int, choices: np.ndarray, size: int,
                    positive_indices: list[np.ndarray] = None, **kwargs):
    if positive_indices is not None and n_samples != len(positive_indices):
        raise ValueError('If specified, "to_exclude" must have the same amount of vectors as '
                         'samples that should be generated.')

    if positive_indices is None:
        positive_indices = [None, ] * n_samples

    return np.stack([fn(choices=choices, size=size, positive_indices=te, **kwargs) for te in positive_indices])


def negative_sample_uniform_multiple(n_samples: int, choices: np.ndarray, size: int,
                                     positive_indices: list[np.ndarray] = None):
    return sample_multiple(negative_sample_uniform, n_samples, choices, size, positive_indices)


def negative_sample_popular_multiple(n_samples: int, choices: np.ndarray, size: int,
                                     popularity_distribution: np.ndarray, squashing_factor: float,
                                     positive_indices: list[np.ndarray] = None):
    return sample_multiple(negative_sample_popular, n_samples, choices, size,
                           popularity_distribution=popularity_distribution, squashing_factor=squashing_factor,
                           positive_indices=positive_indices)
