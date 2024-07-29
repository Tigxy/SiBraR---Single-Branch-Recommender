import operator
import functools
import numpy as np
import pandas as pd


def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)


class ReverseCallable():
    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args):
        return self._fn(*args[::-1])


def get_mask(df, col, op, val):
    # handle special operators
    if op == 'in':
        return df[col].isin(val)
    if op == 'not in':
        return ~get_mask(df, col, 'in', val)

    ops = {
        'eq': operator.eq, 'neq': operator.ne, 'gt': operator.gt, 'ge': operator.ge, 'lt': operator.lt,
        'le': operator.le,
        '=': operator.eq, '!=': operator.ne, '>': operator.gt, '>=': operator.ge, '<': operator.lt, '<=': operator.le
    }
    return ops[op](df[col], val)


def filter_by(df: pd.DataFrame, conditions: list) -> pd.DataFrame:
    if len(conditions) == 0:
        return df
    if len(conditions) == 1:
        return df[get_mask(df, *conditions[0])]
    return df[conjunction(*[get_mask(df, *condition) for condition in conditions])]


pd.DataFrame.filter_by = filter_by
