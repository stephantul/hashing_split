import logging
import struct
from collections.abc import Iterable
from functools import partial
from typing import Any, Callable, Dict, Hashable, Type

logger = logging.getLogger("compute_hash")

# This dictionary contains a mapping from types to a function that converts
# it to something hashable.
# We default to doubles for floats and integers.
# This can lead to loss of information for really really big integers.
ConversionFunction = Callable[[Any], Hashable]
Conversions = Dict[Type, ConversionFunction]


CONVERSIONS: Conversions = {float: partial(struct.pack, "!d")}
BASIC_HASHABLE = tuple([str, bytes])


def _is_iterable(X: Any) -> bool:
    """
    Check if an item is iterable.

    Contains a safeguard for bytes and strings.
    Those are iterable, but can be hashed directly.

    :param X: any object.
    :returns: True if object is iterable.

    """
    if isinstance(X, BASIC_HASHABLE):
        return False
    return isinstance(X, Iterable)


def _recursive_step(X: Any, seed: int, conversions: Conversions) -> int:
    """
    A single step in the hashing algorithm

    :param X: Anything you'd like to hash.
    :param seed: The seed to hash with. This seed is appended to X before hashing.
    :returns: the hash value as a signed 64-bit integer.

    """
    if isinstance(X, BASIC_HASHABLE):
        return hash(X)
    if _is_iterable(X):
        accumulator = 0
        if isinstance(X, dict):
            X = X.items()
        for item in X:
            accumulator ^= recursive_convert(item, seed, conversions)

        return accumulator

    for type_to_check, converter in conversions.items():
        if isinstance(X, type_to_check):
            X_converted = converter(X)
            break
    else:
        X_converted = X

    return hash((X_converted, seed))


def recursive_convert(X: Any, seed: int, additional_conversions: Conversions) -> int:
    """Recursively hash a"""
    actual_conversions = {**CONVERSIONS, **additional_conversions}
    return _recursive_step(X, seed, actual_conversions)
