# MatLibrary (English Documentation)

## 1. Overview

`matlibrary.py` is a unified educational matrix library centered around one `Matrix` class.

It merges and refactors multiple prior implementations into one coherent architecture with:

- A consistent `data/dim` matrix representation.
- Two numeric modes: exact rational (`Fraction`) and numeric floating-point (`float`).
- Core matrix operations and slicing.
- Classical decompositions and advanced linear algebra routines.
- Method-level notes on algorithm quality and stability.

This library is designed for **learning**, **algorithm transparency**, and **small-to-medium scale experiments**, not for high-performance production workloads.

---

## 2. Design Goals

The implementation follows these goals:

1. **Single source of truth**
   - All methods are consolidated in one `Matrix` class.
2. **Architecture consistency**
   - Keep `data/dim` style similar to the earlier `matrix.py` style.
3. **Numerical + exact support**
   - `dtype="fraction"` for exact arithmetic where possible.
   - `dtype="float"` for numerical methods requiring square roots, orthogonal transforms, iterative convergence, etc.
4. **Algorithmic comparability**
   - Multiple methods for the same task where meaningful (for example QR and eigenvalue pipelines).
5. **Educational readability**
   - Implement methods directly from textbook formulas where practical.

---

## 3. Architecture

## 3.1 Core Data Model

Each matrix instance stores:

- `self.data`: nested list representing matrix entries.
- `self.dim`: `(rows, cols)` tuple.
- `self.dtype`: `"fraction"` or `"float"`.
- `self.tolerance`: floating threshold for near-zero tests in numerical routines.

Constructor:

```python
Matrix(data=None, dim=None, init_value=0, tolerance=1e-10, dtype="fraction")
```

Creation patterns:

- From nested list data.
- From shape tuple `dim` and fill value `init_value`.
- From another `Matrix` (deep copied).

Validation includes shape consistency, numeric types, and non-negative dimensions.

## 3.2 Numeric Modes

### Fraction mode (`dtype="fraction"`)

- Stores values as `fractions.Fraction`.
- Useful for exact symbolic-like arithmetic in:
  - addition, subtraction, multiplication,
  - Gaussian elimination (algebraic exactness),
  - determinant, rank, and exact-oriented routines.

### Float mode (`dtype="float"`)

- Stores values as `float`.
- Required for algorithms involving:
  - square roots,
  - iterative convergence,
  - orthogonalization and orthogonal transforms,
  - SVD and Jacobi eigen decomposition.

### Why both modes?

Because linear algebra has two different computational regimes:

- Exact arithmetic is ideal for algebraic derivations and rational matrices.
- Numerical floating arithmetic is necessary for most real-world decomposition algorithms.

## 3.3 Internal Utility Layer

The class includes helper methods:

- `_cast_value`:
  - Converts input values to either `Fraction` or `float` according to `dtype`.
- `_near_zero(value, tol)`:
  - Robust zero test for both exact and floating values.
- `_dot`, `_norm`, `_normalize_vec`:
  - Vector primitives.
- `_mat_vec_mul`:
  - Matrix-vector product helper.
- `_complete_orthonormal_basis`:
  - Completes an orthonormal basis in SVD when rank-deficient.
- `_is_symmetric`:
  - Symmetry test with tolerance.
- `to_float_data`:
  - Converts internal storage to float nested list.

---

## 4. Mathematical Foundations by Method Family

## 4.1 Basic Algebra

- `__add__`, `__sub__`: elementwise operations for same shape matrices.
- `__mul__`:
  - scalar multiplication, or
  - matrix multiplication when dimensions match.
- `__matmul__`: alias to matrix multiplication.
- `__pow__`: fast exponentiation by repeated squaring for square matrices.

## 4.2 Indexing and Slicing

- `__getitem__` supports `(int|slice, int|slice)`.
- `__setitem__` supports scalar assignment and submatrix assignment with shape checks.

This gives NumPy-like matrix slicing ergonomics in pure Python class style.

## 4.3 Gaussian Elimination

Method:

```python
gauss_elimination(pivoting=True) -> (row_echelon_matrix, swap_count)
```

Core idea:

- Row-reduce matrix into row-echelon form.
- Optional partial pivoting improves numerical stability.

Why pivoting is better:

- Reduces catastrophic cancellation.
- Avoids division by very small pivots where possible.

This method is used as a base for determinant and rank.

## 4.4 Determinant

Method:

```python
determinant()
```

Formula used:

- Compute row echelon form.
- Determinant equals product of diagonal entries times $(-1)^{\text{swap\_count}}$.

## 4.5 Inverse (Gauss-Jordan)

Method:

```python
inverse()
```

Approach:

1. Augment matrix with identity: `[A | I]`.
2. Perform Gauss-Jordan elimination with pivoting.
3. Reduce left block to identity.
4. Right block becomes $A^{-1}$.

## 4.6 Rank

Method:

```python
rank()
```

Approach:

- Row-reduce.
- Count non-zero rows (with tolerance handling for float mode).

## 4.7 LU Decomposition (Doolittle)

Method:

```python
lu_decomposition(pivoting=True)
```

Returns:

- With pivoting: `(P, L, U)` such that $PA = LU$.
- Without pivoting: `(L, U)`.

Why pivoting option exists:

- Non-pivot LU may fail for valid matrices with zero/near-zero pivots.
- Pivoted LU is generally robust.

## 4.8 Cholesky Decomposition

Method:

```python
cholesky_decomposition() -> (L, L.T)
```

Requirements:

- Matrix must be square, symmetric, and positive definite (SPD).

Theory:

- For SPD matrix $A$, there exists lower triangular $L$ with
  $$A = LL^T.$$

## 4.9 Gram-Schmidt and QR

Methods:

- `gram_schmidt(method="mgs")`
- `qr_decomposition(method="householder")`

Supported QR styles:

1. **Householder** (recommended)
   - Most numerically stable among provided options.
2. **Modified Gram-Schmidt (MGS)**
   - Better than CGS in finite precision.
3. **Classical Gram-Schmidt (CGS)**
   - Conceptually simple but less stable.

## 4.10 Eigenvalue Methods

### Jacobi Eigenpairs (Symmetric)

Method:

```python
eigenpairs_jacobi(max_iterations=1000, sort=True)
```

Theory:

- Iteratively applies orthogonal plane rotations to annihilate largest off-diagonal entries.
- For real symmetric matrices, converges to diagonal form.
- Diagonal entries are eigenvalues, accumulated rotations are eigenvectors.

### QR Iteration Eigenvalues (General)

Method:

```python
eigenvalues_qr(iterations=100, qr_method="householder")
```

Theory:

- Repeatedly computes $A_k = R_k Q_k$ after $A_{k-1}=Q_k R_k$.
- For many matrices, diagonal tends to eigenvalues.

### Unified API

- `eigenpairs(method="auto", ...)`
- `eigenvalues(method="auto", ...)`

`auto` strategy:

- Symmetric matrix: Jacobi.
- Otherwise: QR iteration.

## 4.11 Singular Value Decomposition (SVD)

Method:

```python
singular_value_decomposition() -> (U, Sigma, V_T)
```

Design in this library:

1. Compute $A^T A$.
2. Use Jacobi eigen decomposition of $A^T A$ to get eigenvalues and $V$.
3. Singular values are $\sigma_i = \sqrt{\lambda_i}$.
4. Build left vectors by
   $$u_i = \frac{Av_i}{\sigma_i}.$$
5. Complete orthonormal basis for deficient-rank cases.
6. Assemble full $U, \Sigma, V^T$.

Why this strategy is preferred here:

- Better alignment consistency than independently diagonalizing $AA^T$ and matching vectors post-hoc.

## 4.12 Jordan Normal Form

Method:

```python
jordan_normal_form()
```

Scope:

- Implemented for rational-eigenvalue scenarios.

Pipeline:

1. Faddeev-LeVerrier for characteristic polynomial coefficients.
2. Rational root theorem for exact rational roots.
3. Rank sequence on powers of $(A-\lambda I)$ to infer Jordan block sizes.
4. Construct Jordan matrix.

Limitations:

- If matrix has non-rational eigenvalues, method raises a descriptive error.

---

## 5. API Reference (Interface Index)

Below is a practical interface list.

## 5.1 Construction and Conversion

- `Matrix(data=None, dim=None, init_value=0, tolerance=1e-10, dtype="fraction")`
- `to_float_data()`
- `copy()`
- `shape()`
- `reshape(newdim)`

## 5.2 Structural / Utility

- `transpose()`
- `T()`
- `sum(axis=None)`
- `concatenate(other, index=0)`
- `kronecker_product(other)`
- `Kronecker_product(other)`
- `identity(n, tolerance=1e-10, dtype="fraction")` (classmethod)

## 5.3 Arithmetic / Operators

- `__add__(other)`
- `__sub__(other)`
- `__mul__(other)`
- `__rmul__(other)`
- `__matmul__(other)`
- `__pow__(n)`
- `num_product(n)`
- `__getitem__(key)`
- `__setitem__(key, value)`
- `__len__()`
- `__str__()`

## 5.4 Core Linear Algebra

- `gauss_elimination(pivoting=True)`
- `determinant()`
- `det()`
- `inverse()`
- `rank()`

## 5.5 Decompositions

- `lu_decomposition(pivoting=True)`
- `cholesky_decomposition()`
- `gram_schmidt(method="mgs")`
- `qr_decomposition(method="householder")`
- `QR_decomposition(method="householder")`

## 5.6 Eigen / SVD / Jordan

- `eigenpairs_jacobi(max_iterations=1000, sort=True)`
- `eigenvalues_qr(iterations=100, qr_method="householder")`
- `eigenpairs(method="auto", iterations=100, max_iterations=1000, qr_method="householder")`
- `eigenvalues(method="auto", iterations=100, max_iterations=1000, qr_method="householder")`
- `singular_value_decomposition()`
- `svd()`
- `jordan_normal_form()`

## 5.7 Global Helpers

- `I(n, tolerance=1e-10, dtype="fraction")`
- `concatenate(items, axis=0)`

---

## 6. Algorithm Comparison Summary

Use:

```python
Matrix.method_comparison()
```

to get markdown text of recommendations.

Main practical defaults:

- Gaussian elimination: pivoting enabled.
- QR: Householder default.
- Eigenvalue pipeline: auto choose Jacobi for symmetric, QR for general.
- SVD: $A^TA$ + vector construction route.

---

## 7. Complexity (High-Level)

Let matrix size be $n \times n$ unless otherwise stated.

- Matrix multiplication: $O(n^3)$
- Gaussian elimination: $O(n^3)$
- Determinant (via elimination): $O(n^3)$
- Inverse (Gauss-Jordan): $O(n^3)$
- LU / Cholesky / QR (dense): about $O(n^3)$
- Jacobi eigen decomposition: typically $O(n^3)$ per sweep scale and iteration dependent
- QR iteration eigenvalues: roughly cubic per iteration for dense matrices
- SVD (current educational implementation): dominated by eigen + matrix-vector operations

---

## 8. Usage Examples

## 8.1 Basic Algebra

```python
from matlibrary import Matrix

A = Matrix(data=[[1, 2], [3, 4]], dtype="fraction")
B = Matrix(data=[[5, 6], [7, 8]], dtype="fraction")

C = A + B
D = A * B
print(C)
print(D)
```

## 8.2 LU and Inverse

```python
A = Matrix(data=[[0, 2], [1, 3]], dtype="fraction")
P, L, U = A.lu_decomposition(pivoting=True)
A_inv = A.inverse()
```

## 8.3 Cholesky

```python
A = Matrix(data=[[4, 2], [2, 3]], dtype="float")
L, LT = A.cholesky_decomposition()
```

## 8.4 Eigenvalues

```python
A = Matrix(data=[[2, 1], [1, 2]], dtype="float")
vals, vecs = A.eigenpairs(method="jacobi")

B = Matrix(data=[[1, 2], [3, 4]], dtype="float")
vals_general = B.eigenvalues(method="qr", iterations=200)
```

## 8.5 SVD

```python
A = Matrix(data=[[3, 2, 2], [2, 3, -2]], dtype="float")
U, Sigma, V_T = A.svd()
A_recon = U * Sigma * V_T
```

---

## 9. Precision and Mode Recommendations

1. Prefer `dtype="fraction"` for exact symbolic-style homework checks.
2. Prefer `dtype="float"` for decomposition-heavy routines.
3. Keep a realistic `tolerance` (for example `1e-10` to `1e-12`) for floating workflows.
4. For very ill-conditioned matrices, expect numerical sensitivity regardless of implementation.

---

## 10. Limitations

1. This is pure Python and not optimized for large dense matrices.
2. Jordan normal form only supports rational-eigenvalue exact path.
3. QR eigen iteration is unshifted educational variant.
4. SVD is educational and explicit, not as robust as LAPACK-grade libraries on all edge cases.

---

## 11. Extension Ideas

1. Add shifted QR / implicit QR for improved eigen convergence.
2. Add Householder bidiagonalization based SVD.
3. Add sparse matrix backend.
4. Add condition number and norm APIs.
5. Add unit test suite with deterministic precision checks.

---

## 12. Quick Interface Cheat Sheet

- Exact mode constructor:
  - `Matrix(data=..., dtype="fraction")`
- Numeric mode constructor:
  - `Matrix(data=..., dtype="float")`
- Core decompose methods:
  - `lu_decomposition`, `cholesky_decomposition`, `qr_decomposition`, `eigenpairs`, `svd`, `jordan_normal_form`
- Global helpers:
  - `I`, `concatenate`

This file aims to be both a usable small matrix toolkit and a readable algorithmic reference.