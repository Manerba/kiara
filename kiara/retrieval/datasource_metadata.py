"""Comparisons for authoritative datasource membership projections."""


def same_ds_ids(actual, expected: list[int]) -> bool:
    """Only a complete, well-typed payload proves that a write is redundant."""
    return (
        isinstance(actual, list)
        and all(type(value) is int for value in actual)
        and sorted(actual) == sorted(expected)
    )
