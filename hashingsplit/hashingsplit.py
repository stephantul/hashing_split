from typing import Hashable, Iterable, List, Optional, Tuple, TypeVar

from hashingsplit.compute_hash import Conversions, recursive_convert

AnyHashable = TypeVar("AnyHashable", bound=object)
Split = Tuple[List[AnyHashable], List[AnyHashable], List[Hashable], List[Hashable]]


def hash_split(
    X: Iterable[AnyHashable],
    y: Iterable[Hashable],
    test_size: float = 0.1,
    seed: int = 0,
    additional_conversions: Optional[Conversions] = None,
) -> Split:
    if additional_conversions is None:
        conversions = {}
    else:
        conversions = additional_conversions
    # This is the modulo factor we use.
    # For a given test size, 1 / test_size items need to go to the test set.
    # If our hash function is uniform, hash % test_mod == 0 will send
    # approximately test_size items to the test set.
    test_mod = round(1 / test_size)

    X_train, X_test, y_train, y_test = [], [], [], []
    # Strict is set to True here because we accept generators.
    # So we can't check the length beforehand.
    for item, label in zip(X, y, strict=True):
        hashed = recursive_convert(item, seed, conversions)
        if (hashed % test_mod) == 0:
            X_test.append(item)
            y_test.append(label)
        else:
            X_train.append(item)
            y_train.append(label)

    return X_train, X_test, y_train, y_test
