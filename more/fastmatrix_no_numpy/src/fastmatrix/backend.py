from __future__ import annotations

import ctypes
import glob
from array import array
from pathlib import Path
from typing import Iterable, Sequence


class BackendError(RuntimeError):
    pass


class _MatrixHandle(ctypes.c_void_p):
    pass


_INT64 = ctypes.c_int64
_DOUBLE = ctypes.c_double


def _load_library() -> ctypes.CDLL:
    root = Path(__file__).resolve().parent
    patterns = [
        "libfastmatrix_native*.so",
        "fastmatrix_native*.dll",
        "libfastmatrix_native*.dylib",
    ]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(str(root / pattern)))
    if not candidates:
        raise FileNotFoundError(
            "Native C++ backend not found. Install the package with 'pip install .' first."
        )
    return ctypes.CDLL(candidates[0])


_lib = _load_library()

_lib.fm_last_error.restype = ctypes.c_char_p
_lib.fm_hardware_threads.restype = ctypes.c_int

_lib.fm_create_from_flat.argtypes = [ctypes.POINTER(_DOUBLE), _INT64, _INT64]
_lib.fm_create_from_flat.restype = _MatrixHandle

_lib.fm_free.argtypes = [_MatrixHandle]
_lib.fm_free.restype = None

_lib.fm_rows.argtypes = [_MatrixHandle]
_lib.fm_rows.restype = _INT64

_lib.fm_cols.argtypes = [_MatrixHandle]
_lib.fm_cols.restype = _INT64

_lib.fm_copy_to_flat.argtypes = [_MatrixHandle, ctypes.POINTER(_DOUBLE), _INT64]
_lib.fm_copy_to_flat.restype = ctypes.c_int

_lib.fm_clone.argtypes = [_MatrixHandle]
_lib.fm_clone.restype = _MatrixHandle

_lib.fm_add.argtypes = [_MatrixHandle, _MatrixHandle, ctypes.c_int]
_lib.fm_add.restype = _MatrixHandle

_lib.fm_sub.argtypes = [_MatrixHandle, _MatrixHandle, ctypes.c_int]
_lib.fm_sub.restype = _MatrixHandle

_lib.fm_scalar_mul.argtypes = [_MatrixHandle, _DOUBLE, ctypes.c_int]
_lib.fm_scalar_mul.restype = _MatrixHandle

_lib.fm_transpose.argtypes = [_MatrixHandle, ctypes.c_int]
_lib.fm_transpose.restype = _MatrixHandle

_lib.fm_matmul.argtypes = [_MatrixHandle, _MatrixHandle, ctypes.c_int, ctypes.c_int]
_lib.fm_matmul.restype = _MatrixHandle


def _last_error() -> str:
    msg = _lib.fm_last_error()
    return msg.decode("utf-8") if msg else "unknown backend error"


def _require_handle(handle: _MatrixHandle) -> _MatrixHandle:
    if not handle or not int(handle.value or 0):
        raise BackendError(_last_error())
    return handle


def _check_code(code: int) -> None:
    if code != 0:
        raise BackendError(_last_error())


def _flatten_nested(data: Iterable[Iterable[float]]) -> tuple[list[float], int, int]:
    rows_data = [list(row) for row in data]
    rows = len(rows_data)
    if rows == 0:
        return [], 0, 0
    cols = len(rows_data[0])
    for row in rows_data:
        if len(row) != cols:
            raise ValueError("All rows must have the same length")
    flat = [float(x) for row in rows_data for x in row]
    return flat, rows, cols


def _to_c_buffer(flat: Sequence[float]) -> tuple[array, ctypes.Array[_DOUBLE]]:
    buf = array("d", flat)
    ptr_type = _DOUBLE * len(buf)
    cbuf = ptr_type(*buf) if buf else ptr_type()
    return buf, cbuf


def create_matrix(data: Iterable[Iterable[float]]) -> _MatrixHandle:
    flat, rows, cols = _flatten_nested(data)
    _, cbuf = _to_c_buffer(flat)
    ptr = cbuf if len(flat) else None
    return _require_handle(_lib.fm_create_from_flat(ptr, rows, cols))


def clone(handle: _MatrixHandle) -> _MatrixHandle:
    return _require_handle(_lib.fm_clone(handle))


def free(handle: _MatrixHandle | None) -> None:
    if handle and int(handle.value or 0):
        _lib.fm_free(handle)


def shape(handle: _MatrixHandle) -> tuple[int, int]:
    rows = int(_lib.fm_rows(handle))
    cols = int(_lib.fm_cols(handle))
    if rows < 0 or cols < 0:
        raise BackendError(_last_error())
    return rows, cols


def tolist(handle: _MatrixHandle) -> list[list[float]]:
    rows, cols = shape(handle)
    total = rows * cols
    ptr_type = _DOUBLE * total
    cbuf = ptr_type() if total else ptr_type()
    _check_code(_lib.fm_copy_to_flat(handle, cbuf, total))
    flat = list(cbuf)
    return [flat[i * cols : (i + 1) * cols] for i in range(rows)]


def hardware_threads() -> int:
    return int(_lib.fm_hardware_threads())


def add(a: _MatrixHandle, b: _MatrixHandle, threads: int = 0) -> _MatrixHandle:
    return _require_handle(_lib.fm_add(a, b, int(threads)))


def sub(a: _MatrixHandle, b: _MatrixHandle, threads: int = 0) -> _MatrixHandle:
    return _require_handle(_lib.fm_sub(a, b, int(threads)))


def scalar_mul(a: _MatrixHandle, scalar: float, threads: int = 0) -> _MatrixHandle:
    return _require_handle(_lib.fm_scalar_mul(a, float(scalar), int(threads)))


def transpose(a: _MatrixHandle, threads: int = 0) -> _MatrixHandle:
    return _require_handle(_lib.fm_transpose(a, int(threads)))


def matmul(a: _MatrixHandle, b: _MatrixHandle, threads: int = 0, block_size: int = 64) -> _MatrixHandle:
    return _require_handle(_lib.fm_matmul(a, b, int(threads), int(block_size)))
