"""Reusable acquisition-order helpers."""

from __future__ import annotations

from itertools import product

import numpy as np


def _count(n):
    n = int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    return n


def sequential(n: int) -> np.ndarray:
    return np.arange(_count(n), dtype=np.intp)


def reverse(n: int) -> np.ndarray:
    return sequential(n)[::-1].copy()


def interleaved(n: int) -> np.ndarray:
    values = sequential(n)
    return np.concatenate((values[::2], values[1::2]))


def center_out(n: int) -> np.ndarray:
    n = _count(n)
    center = (n - 1) / 2.0
    return np.asarray(sorted(range(n), key=lambda index: (abs(index - center), index)), dtype=np.intp)


def outside_in(n: int) -> np.ndarray:
    return center_out(n)[::-1].copy()


def random_order(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).permutation(_count(n))


def outer_product(**dimensions) -> tuple[dict[str, object], ...]:
    names = tuple(dimensions)
    values = tuple(tuple(value) for value in dimensions.values())
    return tuple(dict(zip(names, combination, strict=True)) for combination in product(*values))


def chunk_indices(indices: list[int], size: int) -> list[list[int]]:
    size = max(1, int(size))
    return [indices[i : i + size] for i in range(0, len(indices), size)]


def linear_order(n: int, etl: int) -> list[list[int]]:
    return chunk_indices(list(range(n)), etl)


def outer_inner_order(outer_indices: list[int], inner_len: int) -> list[list[tuple[int, int]]]:
    return [[(outer, inner) for inner in range(inner_len)] for outer in outer_indices]
