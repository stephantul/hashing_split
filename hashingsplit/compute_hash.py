import logging
import struct
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union

from murmurhash import hash as murmurhash

logger = logging.getLogger("compute_hash")

ConversionFunction = Callable[[Any], bytes]
Conversions = Dict[Union[Type, Tuple[Type, ...]], ConversionFunction]


# This dictionary contains a mapping from types to a function that converts
# it to something hashable.
# We default to doubles for floats and integers.
# This can lead to loss of information for really really big integers.
CONVERSIONS: Conversions = {
    (float, int): partial(struct.pack, "!d"),
    str: partial(bytes, encoding="utf-8"),
}
BASIC_HASHABLE = tuple([str, bytes])


def _hashing_function(X: bytes) -> int:
    return murmurhash(X)


def flatten(X: Any) -> List[Any]:
    o = []
    if isinstance(X, BASIC_HASHABLE) or not _is_iterable(X):
        return [X]
    if isinstance(X, dict):
        X = sorted(X.items(), key=lambda x: x[0])
    if isinstance(X, set):
        X = sorted(X)
    for x in X:
        o.extend(flatten(x))
    return o


def _is_iterable(X: Any) -> bool:
    """
    Check if an item is iterable.

    Contains a safeguard for bytes and strings.
    Those are iterable, but can be hashed directly.

    :param X: any object.
    :returns: True if object is iterable.

    """
    try:
        iter(X)
        return True
    except TypeError:
        return False


def _recursive_step(X: Any, seed: int, conversions: Conversions) -> int:
    """
    A single step in the hashing algorithm

    :param X: Anything you'd like to hash.
    :param seed: The seed to hash with. This seed is appended to X before hashing.
    :returns: the hash value as a signed 64-bit integer.

    """
    accumulator = 0
    for x in flatten(X):
        if isinstance(x, BASIC_HASHABLE):
            accumulator ^= _hashing_function(x)
            continue
        for type_to_check, converter in conversions.items():
            if isinstance(x, type_to_check):
                x_converted = converter(x)
                break
        else:
            x_converted = x
        accumulator ^= _hashing_function(x_converted)

    return accumulator


def recursive_convert(X: Any, seed: int, additional_conversions: Conversions) -> int:
    """Recursively hash an object.

    :param X: Anything you'd like to hash.
    :param seed: The seed to hash with. This seed is appended to X before hashing.
    :param additional_conversions: A dictionary mapping from types to functions.
        For each type in additional_conversions, we check if the item in question is an
        instance of that type. If this is the case, we apply the specified conversion.
        Note that it is never necessary to specify conversions for iterables that
        contain primitives, e.g., numpy arrays.
    :returns: the hash value as a signed 64-bit integer.

    """
    actual_conversions = {**CONVERSIONS, **additional_conversions}
    return _recursive_step(X, seed, actual_conversions)
