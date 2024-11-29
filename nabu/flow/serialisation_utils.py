import inspect
from enum import Enum, auto
from functools import wraps
from collections.abc import Callable

from flowjax.bijections import AbstractBijection
from jax.numpy import ndarray

from ._flow_likelihood import FlowLikelihood

__all__ = ["serialise_wrapper"]


def __dir__():
    return __all__


# pylint: disable=protected-access


class ArgumentType(Enum):
    REQUIRED = auto()
    OPTIONAL = auto()
    AUTOSET = auto()

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


class UnsupportedMethod(Exception):
    """Unsupported Method Type"""


def serialise_method(method: Callable) -> dict:
    """
    _summary_

    Args:
        method (``callable``): _description_

    Raises:
        ``UnsupportedMethod``: _description_

    Returns:
        ``dict``:
        _description_
    """
    if isinstance(method, (int, float, type(None), list, tuple)):
        return method
    identifier = method.__name__
    if identifier == "<lambda>":
        raise UnsupportedMethod("Lambda functions are currently not supported")
    args_kwargs = {}
    for key, item in inspect.signature(method).parameters.items():
        if key in ["self", "key"]:
            continue
        if item.default == inspect._empty:
            args_kwargs[key] = ArgumentType.REQUIRED
        elif inspect.isclass(item.default) or inspect.isfunction(item.default):
            args_kwargs[key] = serialise_method(item.default)
        elif callable(item.default):
            args_kwargs[key] = item.default.__name__
        else:
            args_kwargs[key] = item.default
    return {identifier: args_kwargs}


def serialise_wrapper(method: Callable):
    """
    _summary_

    Args:
        method (``Callable``): _description_

    Raises:
        ``ValueError``: _description_

    Returns:
        ``_type_``:
        _description_
    """
    serialised_method = serialise_method(method)
    method_name = method.__name__

    @wraps(method)
    def wrapper(*args, **kwargs):
        for idx, key in enumerate(serialised_method[method_name]):
            if idx < len(args):
                item = args[idx]
            else:
                item = kwargs.get(key, serialised_method[method_name][key])

            if key == "transformer" and isinstance(item, AbstractBijection):
                signature = serialise_method(item.__class__)
                class_structure = vars(item)
                for child_key in signature[item.__class__.__name__].keys():
                    if child_key in class_structure:
                        signature[item.__class__.__name__][child_key] = class_structure[
                            child_key
                        ]
                serialised_method[method_name][key] = signature
            elif key not in serialised_method[method_name]:
                raise UnsupportedMethod(f"invalid argument: {key}")
            elif isinstance(item, (int, str, bool, ndarray, list, tuple, float)):
                serialised_method[method_name][key] = item
            else:
                serialised_method[method_name][key] = serialise_method(item)

        return FlowLikelihood(model=method(*args, **kwargs), metadata=serialised_method)

    wrapper.__annotations__["return"] = FlowLikelihood
    return wrapper
