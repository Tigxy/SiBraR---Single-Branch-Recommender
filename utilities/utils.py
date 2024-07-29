import math
import random
import logging
import functools
import numpy as np
from datetime import datetime

import torch


def generate_id(prefix=None, postfix=None):
    dateTimeObj = datetime.now()
    uid = '{}-{}-{}_{}-{}-{}.{}'.format(dateTimeObj.year, dateTimeObj.month, dateTimeObj.day, dateTimeObj.hour,
                                        dateTimeObj.minute, dateTimeObj.second, dateTimeObj.microsecond)
    if not prefix is None:
        uid = prefix + "_" + uid
    if not postfix is None:
        uid = uid + "_" + postfix
    return uid


def reproducible(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_info_results(metrics):
    """
    Logs the results using logging

    :param metrics: dict containing the metrics to print.
    :return:
    """

    STR_RESULT = "{:10} : {:.5f}"

    for metric_name, metric_value in metrics.items():
        logging.info(STR_RESULT.format(metric_name, metric_value))


class FunctionWrapper:
    """
    Since functions are not properly recognized as enum items, we need to use a wrapper function.
    """

    def __init__(self, function):
        self.function = function
        functools.update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __repr__(self):
        return self.function.__repr__()


def row_wise_sample(a: list, size: int | tuple[int], k: int = 2, replace=False, central_item: any = None, rng=None):
    # transform size to tuple for easier use
    if isinstance(size, int):
        size = (size,)
    output_size = size + (k,)

    if central_item is None:
        choice_fn = rng.choice if rng is not None else np.random.choice
        n_rows = math.prod(size)
        result = np.array([choice_fn(a, k, replace=replace) for i in range(n_rows)])
        return result.reshape(output_size)
    else:
        if central_item not in a:
            raise ValueError(f'central item "{central_item}" must be contained in "a"')

        # need to retrieve proper type for array items, otherwise strings might be cut off
        item_np_type = np.str_(max(a, key=len)).dtype if isinstance(a[0], str) else type(a[0])

        # instantiate container to hold the result
        result = np.empty(shape=output_size, dtype=item_np_type)

        # central modality must always be in list
        result[..., 0] = central_item

        # determine all other modalities that we should sample
        new_item_list = list(set(a) - {central_item})

        # do the sampling
        result[..., 1:] = row_wise_sample(new_item_list, size=size, k=k - 1,
                                          replace=replace, rng=rng)
        return result
