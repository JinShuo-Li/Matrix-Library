# fastmatrix

`fastmatrix` is a **zero-NumPy** matrix package.

It keeps the user-facing API in Python, but moves both **matrix storage** and **dense numeric kernels** into a native C++ shared library. The package is designed for exactly this target architecture:

- Python provides the clean interface
- C++ owns the matrix memory and heavy computation
- `pip install .` works from source
- multithreaded matrix multiplication is built in
- NumPy is not a runtime dependency

## Features

- No `numpy` dependency at install time or runtime
- Native C++ matrix storage (not `numpy.ndarray`)
- Pythonic operators:
  - `A + B`
  - `A - B`
  - `A @ B`
  - `2 * A`
  - `A.T`
- Multithreaded matrix multiplication
- Tunable `threads` and `block_size`
- Standard source install with:

```bash
pip install .
```

## Architecture

This version uses:

- a **C++ shared library** for matrix storage and kernels
- a **Python wrapper layer** for object-oriented usage
- `ctypes` for the bridge layer

That means the design is now:

- C++ owns the matrix buffer
- C++ performs add / sub / transpose / scalar multiply / matmul
- Python is only the API layer

So this is no longer:

- Python data layer + C++ operator layer

Instead it is:

- **Python frontend + C++ data layer + C++ compute layer**

## Why this version removes NumPy entirely

Earlier designs used a Python `numpy.ndarray` as the storage layer and let C++ handle only a subset of operations. That is convenient, but it is **not** a true “Python frontend + C++ backend” architecture.

This version fixes that by making C++ responsible for:

- matrix memory layout
- matrix data ownership
- transpose
- addition / subtraction
- scalar multiplication
- matrix multiplication

Python now just orchestrates the calls.

## Project layout

```text
fastmatrix_no_numpy/
├── pyproject.toml
├── setup.py
├── README.md
├── example.py
├── src/
│   └── fastmatrix/
│       ├── __init__.py
│       ├── backend.py
│       ├── matrix.py
│       └── fastmatrix_native.cpp
└── tests/
    └── test_basic.py
```

## Installation

### Standard install

```bash
pip install .
```

### Development install

```bash
pip install -e .
```

## Windows build notes

On Windows, install **Microsoft C++ Build Tools** first.

Typical command:

```powershell
py -m pip install .
```

The build script compiles a native shared library into the package directory during installation.

Compiler flags configured in `setup.py`:

- Windows: `/O2 /std:c++17 /EHsc`
- Linux / macOS: `-O3 -std=c++17 -fPIC -pthread`

## Quick start

```python
from fastmatrix import Matrix

A = Matrix([[1, 2, 3], [4, 5, 6]])
B = Matrix([[7, 8], [9, 10], [11, 12]])

C = A @ B
print(C.tolist())
# [[58.0, 64.0], [139.0, 154.0]]

print(A.T.tolist())
# [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

print((2 * A).tolist())
# [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]
```

## API

### `Matrix(data)`

Create a dense matrix from nested Python iterables:

```python
A = Matrix([[1, 2], [3, 4]])
```

All rows must have the same length.

### Properties

- `A.shape` → `(rows, cols)`
- `A.T` → transpose as a new `Matrix`

### Methods

- `A.tolist()`
- `A.copy()`
- `A.add(B, threads=0)`
- `A.sub(B, threads=0)`
- `A.matmul(B, threads=0, block_size=64)`

`threads=0` means “auto-detect hardware thread count”.

### Module-level helpers

```python
from fastmatrix import add, sub, transpose, matmul, hardware_threads
```

## Multithreading strategy

The matrix multiplication kernel uses:

1. transposed copy of the right-hand matrix
2. cache-friendly blocking
3. row-range partitioning across worker threads
4. disjoint output writes to avoid write contention

## Validation status

In this environment I verified:

- `pip install .`
- `example.py`
- `pytest`

## Limitations

Current scope is intentionally compact:

- dense matrices only
- numeric conversion through Python `float`
- no slicing yet
- no LU / QR / inverse / SVD yet
- Windows source install is supported by the build script, but I can only runtime-verify Linux in the current environment

## Next step

Natural upgrades for this no-NumPy native design are:

- determinant
- inverse
- LU / QR decomposition
- SIMD optimization
- OpenMP backend switch
- Windows release wheels
