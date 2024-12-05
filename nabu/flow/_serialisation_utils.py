import inspect
from enum import Enum, auto
from functools import wraps, update_wrapper
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


def _get_name(method):
    try:
        return method.__name__
    except AttributeError:
        return type(method).__name__


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
    identifier = _get_name(method)
    if identifier == "<lambda>":
        raise UnsupportedMethod("Lambda functions are currently not supported")
    args_kwargs = {}

    try:
        signature = inspect.signature(method)
    except TypeError:
        signature = inspect.signature(method.__class__)

    for key, item in signature.parameters.items():
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


class BijectorWrapper:
    """Bijection Wrapper"""

    def __init__(self, bijector: AbstractBijection):
        self.bijector = bijector
        update_wrapper(self, bijector)
        # Store the bijector object
        self._args = []
        self._kwargs = {}
        self._executed = False
        self._meta = self.serialise(bijector=bijector)

    def to_dict(self) -> dict:
        """Convert Bijection to dictionary"""
        return self._meta

    @staticmethod
    def serialise(bijector, *args, **kwargs):
        """Serialise underlying bijector"""
        serialised_bij = serialise_method(bijector)
        bij_name = bijector.__name__

        if len(args) != 0 or len(kwargs) != 0:
            for idx, key in enumerate(serialised_bij[bij_name].keys()):
                if idx < len(args):
                    item = args[idx]
                else:
                    item = kwargs.get(key, serialised_bij[bij_name][key])
                    assert item != ArgumentType.REQUIRED, f"Argument `{key}` is missing."

                serialised_bij[bij_name][key] = item
        return serialised_bij

    def __call__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._meta = self.serialise(self.bijector, *args, **kwargs)
        return self

    def execute(self):
        """Execute the bijector with given arguments"""
        return self.bijector(*self._args, **self._kwargs)


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
    method_name = _get_name(method)

    @wraps(method)
    def wrapper(*args, **kwargs):
        processed_args = []
        processed_kwargs = {}
        for idx, key in enumerate(serialised_method[method_name]):
            if idx < len(args):
                item = args[idx]
            else:
                item = kwargs.get(key, serialised_method[method_name][key])

            if isinstance(item, AbstractBijection):
                raise NotImplementedError(
                    "Bijectors has to be wrapped with `bijection_wrapper`"
                )

            if isinstance(item, BijectorWrapper):
                serialised_method[method_name][key] = item.to_dict()
                item = item.execute()
            elif key not in serialised_method[method_name]:
                raise UnsupportedMethod(f"invalid argument: {key}")
            elif isinstance(item, (int, str, bool, ndarray, list, tuple, float)):
                serialised_method[method_name][key] = item
            else:
                serialised_method[method_name][key] = serialise_method(item)

            if idx < len(args):
                processed_args.append(item)
            else:
                processed_kwargs.update({key: item})

        return FlowLikelihood(
            model=method(*processed_args, **processed_kwargs), metadata=serialised_method
        )

    wrapper.__annotations__["return"] = FlowLikelihood
    return wrapper
