from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from . import backend


@dataclass(slots=True)
class Matrix:
    _handle: backend._MatrixHandle

    def __init__(self, data: Iterable[Iterable[float]] | "Matrix", *, _handle: backend._MatrixHandle | None = None):
        if _handle is not None:
            self._handle = _handle
        elif isinstance(data, Matrix):
            self._handle = backend.clone(data._handle)
        else:
            self._handle = backend.create_matrix(data)

    @classmethod
    def _from_handle(cls, handle: backend._MatrixHandle) -> "Matrix":
        obj = cls.__new__(cls)
        obj._handle = handle
        return obj

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            backend.free(handle)
            self._handle = None  # type: ignore[assignment]

    @property
    def shape(self) -> tuple[int, int]:
        return backend.shape(self._handle)

    @property
    def T(self) -> "Matrix":
        return Matrix._from_handle(backend.transpose(self._handle))

    def copy(self) -> "Matrix":
        return Matrix._from_handle(backend.clone(self._handle))

    def tolist(self) -> list[list[float]]:
        return backend.tolist(self._handle)

    def add(self, other: "Matrix", threads: int = 0) -> "Matrix":
        _require_matrix(other)
        return Matrix._from_handle(backend.add(self._handle, other._handle, threads=threads))

    def sub(self, other: "Matrix", threads: int = 0) -> "Matrix":
        _require_matrix(other)
        return Matrix._from_handle(backend.sub(self._handle, other._handle, threads=threads))

    def matmul(self, other: "Matrix", threads: int = 0, block_size: int = 64) -> "Matrix":
        _require_matrix(other)
        return Matrix._from_handle(
            backend.matmul(self._handle, other._handle, threads=threads, block_size=block_size)
        )

    def __add__(self, other: "Matrix") -> "Matrix":
        return self.add(other)

    def __sub__(self, other: "Matrix") -> "Matrix":
        return self.sub(other)

    def __matmul__(self, other: "Matrix") -> "Matrix":
        return self.matmul(other)

    def __mul__(self, scalar: float) -> "Matrix":
        return Matrix._from_handle(backend.scalar_mul(self._handle, float(scalar)))

    def __rmul__(self, scalar: float) -> "Matrix":
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        return f"Matrix(shape={self.shape}, data={self.tolist()})"


def _require_matrix(obj: object) -> None:
    if not isinstance(obj, Matrix):
        raise TypeError("operand must be a Matrix")
