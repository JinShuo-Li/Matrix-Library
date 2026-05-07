from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer


def _lib_name() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "fastmatrix.dll"
    if system == "darwin":
        return "libfastmatrix.dylib"
    return "libfastmatrix.so"


_LIB_PATH = Path(__file__).resolve().parent / _lib_name()
if not _LIB_PATH.exists():
    raise FileNotFoundError(
        f"C++ backend not found: {_LIB_PATH}. Please run build.sh first."
    )

_lib = ctypes.CDLL(os.fspath(_LIB_PATH))
_double_2d = ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")

_lib.fm_last_error.restype = ctypes.c_char_p
_lib.fm_hardware_threads.restype = ctypes.c_int

_lib.fm_add.argtypes = [_double_2d, _double_2d, _double_2d, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
_lib.fm_add.restype = ctypes.c_int

_lib.fm_sub.argtypes = [_double_2d, _double_2d, _double_2d, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
_lib.fm_sub.restype = ctypes.c_int

_lib.fm_transpose.argtypes = [_double_2d, _double_2d, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
_lib.fm_transpose.restype = ctypes.c_int

_lib.fm_matmul.argtypes = [
    _double_2d,
    _double_2d,
    _double_2d,
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


def as_2d_float64(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("input must be a 2D matrix")
    return np.ascontiguousarray(arr)


def hardware_threads() -> int:
    return int(_lib.fm_hardware_threads())


def add(a, b, threads: int = 0) -> np.ndarray:
    a = as_2d_float64(a)
    b = as_2d_float64(b)
    if a.shape != b.shape:
        raise ValueError("add shape mismatch")
    out = np.empty_like(a)
    _check(_lib.fm_add(a, b, out, a.shape[0], a.shape[1], threads))
    return out


def sub(a, b, threads: int = 0) -> np.ndarray:
    a = as_2d_float64(a)
    b = as_2d_float64(b)
    if a.shape != b.shape:
        raise ValueError("sub shape mismatch")
    out = np.empty_like(a)
    _check(_lib.fm_sub(a, b, out, a.shape[0], a.shape[1], threads))
    return out


def transpose(a, threads: int = 0) -> np.ndarray:
    a = as_2d_float64(a)
    out = np.empty((a.shape[1], a.shape[0]), dtype=np.float64)
    _check(_lib.fm_transpose(a, out, a.shape[0], a.shape[1], threads))
    return out


def matmul(a, b, threads: int = 0, block_size: int = 64) -> np.ndarray:
    a = as_2d_float64(a)
    b = as_2d_float64(b)
    if a.shape[1] != b.shape[0]:
        raise ValueError("matmul dimension mismatch: a.shape[1] must equal b.shape[0]")
    out = np.empty((a.shape[0], b.shape[1]), dtype=np.float64)
    _check(_lib.fm_matmul(a, b, out, a.shape[0], a.shape[1], b.shape[1], threads, block_size))
    return out
