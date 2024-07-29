import inspect
from copy import deepcopy

from mashumaro import DataClassDictMixin
from dataclasses import field as std_field_fn, dataclass, fields, asdict


def excludable_field(default, exclude_from_dict: bool = True, **kwargs):
    metadata = kwargs.get('metadata', {})
    if not isinstance(metadata, dict):
        raise SystemError("Provided metadata for the field must be of type 'dict'")
    metadata['exclude_from_dict'] = exclude_from_dict
    return std_field_fn(default=default, metadata=metadata, **kwargs)


def merge_dicts(first: dict, second: dict):
    """
    Merges two dictionaries and all their subsequent dictionaries.
    In case both dictionaries contain the same key, which is not another dictionary, the latter one is used.

    This merges in contrast to dict.update() all subdicts and its items
    instead of overriding the former with the latter.
    """
    fk = set(first.keys())
    sk = set(second.keys())
    common_keys = fk.intersection(sk)

    z = {}
    for k in common_keys:
        if isinstance(first[k], dict) and isinstance(second[k], dict):
            z[k] = merge_dicts(first[k], second[k])
        else:
            z[k] = deepcopy(second[k])

    for k in fk - common_keys:
        z[k] = deepcopy(first[k])

    for k in sk - common_keys:
        z[k] = deepcopy(second[k])

    return z


@dataclass
class BaseConfig(DataClassDictMixin):
    def as_dict(self):
        valid_param_names = {field.name for field in fields(self) if not field.metadata.get('exclude_from_dict', False)}
        return {k: v for k, v in asdict(self).items() if k in valid_param_names}

    @classmethod
    def from_dict_ext(cls, d: dict, dict_has_precedence: bool = False, **kwargs):
        # unfortunately overriding from_dict() fails as it would not accept any kwargs in subclasses
        # this is probably related to https://github.com/Fatal1ty/mashumaro/issues/78
        # To save time, for now we'll simply be using a different method name

        # override in-code supplied parameters with config parameters (or vice versa)
        # this is to ensure which parameters have priority
        d = merge_dicts(kwargs, d) if dict_has_precedence else merge_dicts(d, kwargs)
        return cls.from_dict(d)

    @classmethod
    def populate_default_values(cls, d: dict, **kwargs):
        return cls.from_dict_ext(d, **kwargs).as_dict()


@dataclass
class SoftBaseConfig(BaseConfig):
    """
    Configuration datclass that overrides the 'from_dict' method to ignore parameters
    that are not listed in the dataclass.
    """

    @classmethod
    def from_dict_ext(cls, d: dict, dict_has_precedence: bool = False, **kwargs):
        # similar to base parent class, merge dict and additional parameters
        d = merge_dicts(kwargs, d) if dict_has_precedence else merge_dicts(d, kwargs)

        # check which parameters are defined, and drop those that are not known
        class_parameters = inspect.signature(cls).parameters
        d = {k: v for k, v in d.items() if k in class_parameters}
        return cls.from_dict(d)
