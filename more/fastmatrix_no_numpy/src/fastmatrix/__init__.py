from .backend import hardware_threads
from .matrix import Matrix

__version__ = "0.3.0"


def add(a: Matrix, b: Matrix, threads: int = 0) -> Matrix:
    return a.add(b, threads=threads)


def sub(a: Matrix, b: Matrix, threads: int = 0) -> Matrix:
    return a.sub(b, threads=threads)


def transpose(a: Matrix, threads: int = 0) -> Matrix:
    if threads == 0:
        return a.T
    from . import backend
    return Matrix._from_handle(backend.transpose(a._handle, threads=threads))


def matmul(a: Matrix, b: Matrix, threads: int = 0, block_size: int = 64) -> Matrix:
    return a.matmul(b, threads=threads, block_size=block_size)


__all__ = [
    "Matrix",
    "add",
    "sub",
    "transpose",
    "matmul",
    "hardware_threads",
    "__version__",
]
