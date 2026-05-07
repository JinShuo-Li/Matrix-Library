from __future__ import annotations

import ctypes
import glob
from pathlib import Path
from typing import Any

import numpy as np
from numpy.ctypeslib import ndpointer


_DOUBLE_2D = ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")


def _load_library() -> ctypes.CDLL:
    root = Path(__file__).resolve().parent
    patterns = [
        "_fastmatrix_native*.so",
        "_fastmatrix_native*.pyd",
        "_fastmatrix_native*.dll",
        "_fastmatrix_native*.dylib",
    ]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(str(root / pattern)))
    if not candidates:
        raise FileNotFoundError(
            "C++ backend not found. Run 'pip install .' or 'pip install -e .' first."
        )
    return ctypes.CDLL(candidates[0])


_lib = _load_library()

_lib.fm_last_error.restype = ctypes.c_char_p
_lib.fm_hardware_threads.restype = ctypes.c_int

_lib.fm_add.argtypes = [_DOUBLE_2D, _DOUBLE_2D, _DOUBLE_2D, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
_lib.fm_add.restype = ctypes.c_int

_lib.fm_sub.argtypes = [_DOUBLE_2D, _DOUBLE_2D, _DOUBLE_2D, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
_lib.fm_sub.restype = ctypes.c_int

_lib.fm_transpose.argtypes = [_DOUBLE_2D, _DOUBLE_2D, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
_lib.fm_transpose.restype = ctypes.c_int

_lib.fm_matmul.argtypes = [
    _DOUBLE_2D,
    _DOUBLE_2D,
    _DOUBLE_2D,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int,
    ctypes.c_int,
]
_lib.fm_matmul.restype = ctypes.c_int


def _check(code: int) -> None:
    if code != 0:
        msg = _lib.fm_last_error()
        raise RuntimeError(msg.decode("utf-8") if msg else f"backend error {code}")


def as_2d_float64(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("input must be a 2D matrix")
    return np.ascontiguousarray(arr)


def hardware_threads() -> int:
    return int(_lib.fm_hardware_threads())


def add(a: Any, b: Any, threads: int = 0) -> np.ndarray:
    a = as_2d_float64(a)
    b = as_2d_float64(b)
    if a.shape != b.shape:
        raise ValueError("add shape mismatch")
    out = np.empty_like(a)
    _check(_lib.fm_add(a, b, out, a.shape[0], a.shape[1], int(threads)))
    return out


def sub(a: Any, b: Any, threads: int = 0) -> np.ndarray:
    a = as_2d_float64(a)
    b = as_2d_float64(b)
    if a.shape != b.shape:
        raise ValueError("sub shape mismatch")
    out = np.empty_like(a)
    _check(_lib.fm_sub(a, b, out, a.shape[0], a.shape[1], int(threads)))
    return out


def transpose(a: Any, threads: int = 0) -> np.ndarray:
    a = as_2d_float64(a)
    out = np.empty((a.shape[1], a.shape[0]), dtype=np.float64)
    _check(_lib.fm_transpose(a, out, a.shape[0], a.shape[1], int(threads)))
    return out


def matmul(a: Any, b: Any, threads: int = 0, block_size: int = 64) -> np.ndarray:
    a = as_2d_float64(a)
    b = as_2d_float64(b)
    if a.shape[1] != b.shape[0]:
        raise ValueError("matmul dimension mismatch: a.shape[1] must equal b.shape[0]")
    out = np.empty((a.shape[0], b.shape[1]), dtype=np.float64)
    _check(_lib.fm_matmul(a, b, out, a.shape[0], a.shape[1], b.shape[1], int(threads), int(block_size)))
    return out
