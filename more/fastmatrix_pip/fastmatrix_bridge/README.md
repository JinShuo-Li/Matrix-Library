# fastmatrix

> A Python-first matrix package with a multithreaded C++ backend.

`fastmatrix` is a small matrix library that combines:

- **Python-side ergonomics** for object-oriented usage and easy scripting
- **C++-side performance** for dense numerical kernels
- **Multithreaded matrix multiplication** for CPU-side acceleration
- **`pip install` packaging** for source builds on Windows, Linux, and macOS

This project is designed as a practical bridge between the ease of Python syntax and the execution speed of native C++ code.

---

## Why this project exists

Many educational matrix libraries are excellent for learning linear algebra algorithms, but they are usually not organized around:

1. **a Pythonic front-end API**,
2. **a native C++ compute backend**, and
3. **a packaging workflow that can be installed directly with `pip`**.

`fastmatrix` focuses exactly on that combination.

The current version keeps the public API intentionally small and stable, while moving the heavy kernel for dense matrix operations into C++.

---

## Features

- `Matrix` class with a Python-friendly interface
- Native C++ backend compiled during installation
- Dense matrix operations:
  - addition
  - subtraction
  - transpose
  - matrix multiplication
- Multithreaded matrix multiplication
- Cache-aware blocked multiplication
- Pre-transpose optimization for the right-hand matrix in `matmul`
- Works as a standard Python package via `pip install .`
- Source-build support for:
  - Windows
  - Linux
  - macOS

---

## Architecture

The package is split into two layers.

### 1. Python layer

The Python layer provides:

- the `Matrix` class
- operator overloading such as `@`, `+`, `-`
- NumPy interoperability
- simple user-facing parameter control (`threads`, `block_size`)

### 2. C++ layer

The C++ layer provides the compute kernels:

- `fm_add`
- `fm_sub`
- `fm_transpose`
- `fm_matmul`
- `fm_hardware_threads`

The Python layer calls the compiled native module through `ctypes`.

This gives the project a clean separation:

- Python handles API ergonomics
- C++ handles performance-sensitive computation

---

## Matrix multiplication optimization strategy

The `matmul` kernel uses several practical CPU-side optimizations.

### Pre-transpose of `B`

For `C = A @ B`, the backend first builds `B^T` internally.

This changes the access pattern during the dot product so that both rows are traversed in a cache-friendlier way.

### Blocking

The multiplication uses blocked loops:

- row blocks
- column blocks
- inner-dimension blocks

This improves cache locality and reduces the performance penalty of repeatedly touching large matrices.

### Multithreading

The output matrix is partitioned by row ranges, and each worker thread computes a disjoint subset of rows.

Benefits:

- simple parallel decomposition
- no write conflicts between threads
- straightforward scaling on multi-core CPUs

---

## Project layout

```text
fastmatrix_bridge/
├── pyproject.toml
├── setup.py
├── MANIFEST.in
├── README.md
├── example.py
├── benchmark.py
├── src/
│   └── fastmatrix/
│       ├── __init__.py
│       ├── matrix.py
│       ├── backend.py
│       └── _fastmatrix_native.cpp
└── tests/
    └── test_basic.py
```

---

## Requirements

- Python **3.9+**
- NumPy **1.26+**
- A C++ compiler capable of **C++17**

---

## Installation

### Standard install

```bash
pip install .
```

This is the default installation command for regular users.

### Development install

```bash
pip install -e .
```

Use editable mode only when you are modifying the source code locally.

### Install from source distribution

```bash
pip install dist/fastmatrix-0.2.0.tar.gz
```

---

## Windows installation

On Windows, `pip` will compile the native extension from source.

You need:

- **Microsoft C++ Build Tools**
- a Python environment that matches your compiler toolchain

Recommended steps:

```powershell
py -m pip install --upgrade pip setuptools wheel
py -m pip install .
```

### Notes for Windows users

- The extension is compiled with MSVC flags such as `/O2` and `/std:c++17`
- If compilation fails, the most common reason is that Build Tools are not installed or the C++ workload was not selected
- Install the **Desktop development with C++** workload in Visual Studio Build Tools

---

## Linux installation

Typical command:

```bash
pip install .
```

The extension is compiled with flags similar to:

```bash
-O3 -std=c++17 -pthread
```

On Debian/Ubuntu-like systems, you may need:

```bash
sudo apt-get install build-essential python3-dev
```

---

## macOS installation

Install Xcode Command Line Tools first if needed:

```bash
xcode-select --install
```

Then install the package:

```bash
pip install .
```

---

## Quick start

### Functional API

```python
import numpy as np
from fastmatrix import add, sub, transpose, matmul, hardware_threads

print("hardware threads:", hardware_threads())

a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[5.0, 6.0], [7.0, 8.0]])

print(add(a, b))
print(sub(a, b))
print(transpose(a))
print(matmul(a, b, threads=4, block_size=64))
```

### Object-oriented API

```python
from fastmatrix import Matrix

A = Matrix([[1, 2, 3], [4, 5, 6]])
B = Matrix([[7, 8], [9, 10], [11, 12]])

C = A @ B
print(C)
print(C.numpy())

D = A.matmul(B, threads=4, block_size=64)
print(D.numpy())

print(A.T.numpy())
```

---

## Public API

The package exports:

```python
from fastmatrix import Matrix, add, sub, transpose, matmul, hardware_threads
```

### `Matrix`

A lightweight Python wrapper over a contiguous `float64` 2D NumPy array.

#### Constructor

```python
Matrix(data)
```

Accepted inputs:

- another `Matrix`
- a 2D NumPy array
- a nested Python list or similar 2D iterable

#### Properties and methods

- `shape` → returns `(rows, cols)`
- `T` → returns the transpose as a new `Matrix`
- `numpy(copy=True)` → returns the underlying NumPy array
- `matmul(other, threads=0, block_size=64)` → matrix multiplication

#### Operators

- `A @ B`
- `A + B`
- `A - B`
- `scalar * A`
- `A * scalar`

### `add(a, b, threads=0)`

Elementwise matrix addition.

### `sub(a, b, threads=0)`

Elementwise matrix subtraction.

### `transpose(a, threads=0)`

Matrix transpose.

### `matmul(a, b, threads=0, block_size=64)`

Dense matrix multiplication.

Parameters:

- `threads=0` means: use hardware concurrency when available
- `block_size=64` is the default tiling size for the blocked kernel

### `hardware_threads()`

Returns the number of worker threads the backend would use by default.

---

## Data model and constraints

Current implementation choices:

- data type: **`float64` only**
- array layout: **2D C-contiguous arrays**
- backend input validation is strict

That means:

- non-2D inputs will raise `ValueError`
- incompatible matrix shapes will raise `ValueError`
- `block_size <= 0` will raise a backend error

Example:

```python
from fastmatrix import matmul

# valid
c = matmul([[1, 2]], [[3], [4]])

# invalid: dimension mismatch
# matmul([[1, 2]], [[3, 4]])
```

---

## Benchmarking

A simple benchmark script is included:

```bash
python benchmark.py
```

This can be used to compare:

- different thread counts
- different block sizes
- Python/NumPy-side data preparation cost vs native compute cost

When benchmarking, keep in mind:

- small matrices may not benefit much from multithreading
- larger matrices benefit more from blocking and thread parallelism
- results depend on CPU model, cache hierarchy, and memory bandwidth

---

## Testing

Run tests with:

```bash
pytest
```

Current tests cover:

- add/sub/transpose/matmul correctness
- `Matrix` wrapper behavior

---

## Build details

The package uses:

- `pyproject.toml` for build-system metadata
- `setup.py` to define the C++ extension and platform-specific compile flags

### Compilation flags

#### Windows

```text
/O2 /std:c++17 /EHsc
```

#### Linux/macOS

```text
-O3 -std=c++17 -pthread
```

---

## Design rationale

This project intentionally does **not** try to replace NumPy.

Instead, it demonstrates a clear engineering pattern:

- expose a convenient Python API
- keep the public interface compact
- move compute-intensive kernels into native C++
- package everything in a way that can be installed through standard Python tooling

This makes the project suitable for:

- learning Python/C++ hybrid library design
- experimenting with native numerical kernels
- building a foundation for a larger linear algebra package

---

## Current limitations

At the moment, the project is deliberately minimal.

Limitations include:

- only dense matrices are supported
- only `float64` is supported
- no SIMD intrinsics yet
- no BLAS/OpenMP backend yet
- no advanced linear algebra routines yet
  - LU
  - QR
  - determinant
  - inverse
  - eigen decomposition
  - SVD
- no prebuilt Windows wheel is included in this repository snapshot

---

## Future work

Good next steps include:

1. add more linear algebra kernels in C++
2. support SIMD acceleration
3. optionally add OpenMP or thread-pool based scheduling
4. expose more zero-copy pathways between NumPy and the backend
5. provide wheels for Windows, Linux, and macOS
6. evaluate a `pybind11` version alongside the current `ctypes` bridge

---

## Example workflows

### Regular use

```bash
pip install .
python example.py
```

### Local development

```bash
pip install -e .
pytest
python example.py
python benchmark.py
```

---

## License

This package metadata currently declares an **MIT** license.

If you plan to publish or redistribute the project, make sure a corresponding `LICENSE` file is included in the repository.

---

## Summary

`fastmatrix` is a compact demonstration of a useful hybrid pattern:

- **Python for usability**
- **C++ for computation**
- **multithreading for CPU acceleration**
- **standard packaging for installation**

If you want to extend this project, the most natural direction is to keep the Python API stable and gradually move more linear algebra algorithms into the native backend.
