from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np

from . import backend

ArrayLike = Union["Matrix", np.ndarray, Iterable[Iterable[float]]]


def _unwrap(x: ArrayLike) -> np.ndarray:
    if isinstance(x, Matrix):
        return x.data
    return backend.as_2d_float64(x)


@dataclass
class Matrix:
    data: np.ndarray

    def __init__(self, data: ArrayLike):
        self.data = _unwrap(data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self) -> "Matrix":
        return Matrix(backend.transpose(self.data))

    def numpy(self, copy: bool = True) -> np.ndarray:
        return self.data.copy() if copy else self.data

    def matmul(self, other: ArrayLike, threads: int = 0, block_size: int = 64) -> "Matrix":
        return Matrix(backend.matmul(self.data, _unwrap(other), threads=threads, block_size=block_size))

    def __matmul__(self, other: ArrayLike) -> "Matrix":
        return self.matmul(other)

    def __add__(self, other: ArrayLike) -> "Matrix":
        return Matrix(backend.add(self.data, _unwrap(other)))

    def __sub__(self, other: ArrayLike) -> "Matrix":
        return Matrix(backend.sub(self.data, _unwrap(other)))

    def __mul__(self, scalar: float) -> "Matrix":
        return Matrix(self.data * float(scalar))

    def __rmul__(self, scalar: float) -> "Matrix":
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        return f"Matrix(shape={self.data.shape}, data=\n{self.data})"
