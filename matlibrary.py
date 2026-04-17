import copy
import math
from fractions import Fraction


def _is_number(value):
	return isinstance(value, (int, float, Fraction))

class Matrix:
	"""
	Unified matrix class that consolidates features from:
	- matrix.py (data/dim architecture, indexing, Jordan form, helpers)
	- hw1/cholesky.py (Cholesky, Gram-Schmidt, QR, QR eigenvalues)
	- hw1/incomplete_lu_matrix_class_maybe.py (LU, exact arithmetic utilities)
	- hw2/singular_value_decomposition.py (Jacobi eigenpairs + SVD)

	The class supports two storage modes:
	- dtype="fraction" for exact rational arithmetic where possible
	- dtype="float" for numerical linear algebra
	"""

	def __init__(self, data=None, dim=None, init_value=0, tolerance=1e-10, dtype="fraction"):
		if dtype not in {"fraction", "float"}:
			raise ValueError("dtype must be 'fraction' or 'float'.")

		self.dtype = dtype
		self.tolerance = float(tolerance)

		if data is None and dim is None:
			raise ValueError("Either data or dim must be provided.")

		if isinstance(data, Matrix):
			data = copy.deepcopy(data.data)

		if data is not None:
			if not isinstance(data, list):
				raise TypeError("data must be a nested list or Matrix.")

			if len(data) == 0:
				self.data = []
				self.dim = (0, 0)
				self.init_value = init_value
				return

			row_len = None
			parsed = []
			for row in data:
				if not isinstance(row, list):
					raise TypeError("Every row in data must be a list.")
				if row_len is None:
					row_len = len(row)
				elif len(row) != row_len:
					raise ValueError("All rows in data must have the same length.")

				parsed_row = [self._cast_value(x) for x in row]
				parsed.append(parsed_row)

			self.data = parsed
			self.dim = (len(parsed), row_len)
			self.init_value = init_value
			return

		if not isinstance(dim, tuple) or len(dim) != 2:
			raise TypeError("dim must be a tuple (rows, cols).")
		m, n = dim
		if not (isinstance(m, int) and isinstance(n, int)):
			raise TypeError("dim entries must be integers.")
		if m < 0 or n < 0:
			raise ValueError("dim entries must be non-negative.")

		fill = self._cast_value(init_value)
		self.data = [[fill for _ in range(n)] for _ in range(m)]
		self.dim = (m, n)
		self.init_value = init_value

	def _cast_value(self, value):
		if not _is_number(value):
			raise TypeError("Matrix elements must be int, float, or Fraction.")
		if self.dtype == "fraction":
			return value if isinstance(value, Fraction) else Fraction(value)
		return float(value)

	@staticmethod
	def _near_zero(value, tol):
		if isinstance(value, Fraction):
			return value == 0
		return abs(float(value)) <= tol

	@staticmethod
	def _dot(v1, v2):
		return sum(v1[i] * v2[i] for i in range(len(v1)))

	@classmethod
	def _norm(cls, vec):
		return math.sqrt(max(0.0, cls._dot(vec, vec)))

	@classmethod
	def _normalize_vec(cls, vec, tol):
		nrm = cls._norm(vec)
		if nrm <= tol:
			raise ValueError("Cannot normalize a near-zero vector.")
		return [x / nrm for x in vec]

	@staticmethod
	def _mat_vec_mul(mat, vec):
		return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]

	@classmethod
	def _complete_orthonormal_basis(cls, base_vectors, dimension, tol):
		basis = []
		for vec in base_vectors:
			v = vec[:]
			for q in basis:
				coeff = cls._dot(v, q)
				v = [v[i] - coeff * q[i] for i in range(dimension)]
			nrm = cls._norm(v)
			if nrm > tol:
				basis.append([x / nrm for x in v])

		for idx in range(dimension):
			v = [0.0] * dimension
			v[idx] = 1.0
			for q in basis:
				coeff = cls._dot(v, q)
				v = [v[i] - coeff * q[i] for i in range(dimension)]
			nrm = cls._norm(v)
			if nrm > tol:
				basis.append([x / nrm for x in v])
			if len(basis) == dimension:
				break

		if len(basis) < dimension:
			raise ValueError("Failed to complete an orthonormal basis.")
		return basis

	def _is_symmetric(self):
		m, n = self.dim
		if m != n:
			return False
		for i in range(m):
			for j in range(i + 1, n):
				if abs(float(self.data[i][j] - self.data[j][i])) > self.tolerance:
					return False
		return True

	def to_float_data(self):
		return [[float(x) for x in row] for row in self.data]

	@staticmethod
	def method_comparison():
		"""
		Returns markdown-style notes comparing methods for the same task.
		"""
		return (
			"# Method Comparison\n"
			"- Gaussian elimination: partial pivoting is more stable than no-pivot elimination; use pivoting by default.\n"
			"- QR decomposition: Modified Gram-Schmidt is usually better than Classical Gram-Schmidt in finite precision.\n"
			"- QR with Householder is typically the most numerically stable general-purpose QR.\n"
			"- Eigenvalues: Jacobi is preferred for real symmetric matrices (stable, gives orthonormal eigenvectors).\n"
			"- Eigenvalues: QR iteration is more general for non-symmetric matrices, but eigenvectors are not returned here.\n"
			"- SVD: using A^T A for V and then U = A v_i / sigma_i is more robust than independently matching U from A A^T.\n"
			"- Cholesky is best for symmetric positive-definite matrices; LU is more general."
		)

	def shape(self):
		return self.dim

	def copy(self):
		return Matrix(
			data=copy.deepcopy(self.data),
			tolerance=self.tolerance,
			dtype=self.dtype,
		)

	def reshape(self, newdim):
		if not isinstance(newdim, tuple) or len(newdim) != 2:
			raise TypeError("newdim must be a tuple (rows, cols).")
		nr, nc = newdim
		if not (isinstance(nr, int) and isinstance(nc, int)):
			raise TypeError("newdim entries must be integers.")
		if nr < 0 or nc < 0:
			raise ValueError("newdim entries must be non-negative.")

		r, c = self.dim
		if r * c != nr * nc:
			raise ValueError("Cannot reshape to a different total number of elements.")

		flat = [self.data[i][j] for i in range(r) for j in range(c)]
		idx = 0
		out = []
		for _ in range(nr):
			row = []
			for _ in range(nc):
				row.append(flat[idx])
				idx += 1
			out.append(row)
		return Matrix(out, tolerance=self.tolerance, dtype=self.dtype)

	def transpose(self):
		r, c = self.dim
		if r == 0:
			return Matrix(dim=(c, r), tolerance=self.tolerance, dtype=self.dtype)
		out = [[self.data[i][j] for i in range(r)] for j in range(c)]
		return Matrix(out, tolerance=self.tolerance, dtype=self.dtype)

	def T(self):
		return self.transpose()

	@classmethod
	def identity(cls, n, tolerance=1e-10, dtype="fraction"):
		if not isinstance(n, int) or n < 0:
			raise ValueError("n must be a non-negative integer.")
		one = Fraction(1) if dtype == "fraction" else 1.0
		zero = Fraction(0) if dtype == "fraction" else 0.0
		out = [[one if i == j else zero for j in range(n)] for i in range(n)]
		return cls(out, tolerance=tolerance, dtype=dtype)

	def sum(self, axis=None):
		r, c = self.dim
		if axis not in {None, 0, 1}:
			raise ValueError("axis must be None, 0, or 1.")

		if axis is None:
			total = 0
			for row in self.data:
				for val in row:
					total += val
			return Matrix([[total]], tolerance=self.tolerance, dtype=self.dtype)

		if axis == 0:
			out = [[sum(self.data[i][j] for i in range(r)) for j in range(c)]]
			return Matrix(out, tolerance=self.tolerance, dtype=self.dtype)

		out = [[sum(self.data[i][j] for j in range(c))] for i in range(r)]
		return Matrix(out, tolerance=self.tolerance, dtype=self.dtype)

	def concatenate(self, other, index=0):
		if not isinstance(other, Matrix):
			raise TypeError("other must be a Matrix.")

		out_dtype = "fraction" if self.dtype == other.dtype == "fraction" else "float"
		out_tol = max(self.tolerance, other.tolerance)

		if index == 0:
			if self.dim[1] != other.dim[1]:
				raise ValueError("For row concatenation, column counts must match.")
			data = copy.deepcopy(self.data) + copy.deepcopy(other.data)
			return Matrix(data, tolerance=out_tol, dtype=out_dtype)

		if index == 1:
			if self.dim[0] != other.dim[0]:
				raise ValueError("For column concatenation, row counts must match.")
			data = [
				copy.deepcopy(self.data[i]) + copy.deepcopy(other.data[i])
				for i in range(self.dim[0])
			]
			return Matrix(data, tolerance=out_tol, dtype=out_dtype)

		raise ValueError("index must be 0 (rows) or 1 (cols).")

	def kronecker_product(self, other):
		if not isinstance(other, Matrix):
			raise TypeError("other must be a Matrix.")
		m1, n1 = self.dim
		m2, n2 = other.dim
		out_dtype = "fraction" if self.dtype == other.dtype == "fraction" else "float"
		out_tol = max(self.tolerance, other.tolerance)

		data = [[0 for _ in range(n1 * n2)] for _ in range(m1 * m2)]
		for i in range(m1):
			for j in range(n1):
				for p in range(m2):
					for q in range(n2):
						data[i * m2 + p][j * n2 + q] = self.data[i][j] * other.data[p][q]
		return Matrix(data, tolerance=out_tol, dtype=out_dtype)

	def Kronecker_product(self, other):
		return self.kronecker_product(other)

	def __getitem__(self, key):
		if not isinstance(key, tuple) or len(key) != 2:
			raise TypeError("Index must be a tuple (row_key, col_key).")
		row_key, col_key = key
		r, c = self.dim

		def _resolve(selector, size):
			if isinstance(selector, int):
				idx = selector + size if selector < 0 else selector
				if idx < 0 or idx >= size:
					raise IndexError("Index out of range.")
				return [idx], True
			if isinstance(selector, slice):
				return list(range(*selector.indices(size))), False
			raise TypeError("Each index must be int or slice.")

		row_idx, row_single = _resolve(row_key, r)
		col_idx, col_single = _resolve(col_key, c)

		if row_single and col_single:
			return self.data[row_idx[0]][col_idx[0]]

		if len(row_idx) == 0:
			return Matrix(dim=(0, len(col_idx)), tolerance=self.tolerance, dtype=self.dtype)

		data = [[self.data[i][j] for j in col_idx] for i in row_idx]
		return Matrix(data, tolerance=self.tolerance, dtype=self.dtype)

	def __setitem__(self, key, value):
		if not isinstance(key, tuple) or len(key) != 2:
			raise TypeError("Index must be a tuple (row_key, col_key).")
		row_key, col_key = key
		r, c = self.dim

		def _resolve(selector, size):
			if isinstance(selector, int):
				idx = selector + size if selector < 0 else selector
				if idx < 0 or idx >= size:
					raise IndexError("Index out of range.")
				return [idx], True
			if isinstance(selector, slice):
				return list(range(*selector.indices(size))), False
			raise TypeError("Each index must be int or slice.")

		row_idx, row_single = _resolve(row_key, r)
		col_idx, col_single = _resolve(col_key, c)

		if row_single and col_single:
			if not _is_number(value):
				raise TypeError("Scalar assignment requires a numeric value.")
			self.data[row_idx[0]][col_idx[0]] = self._cast_value(value)
			return

		if not isinstance(value, Matrix):
			raise TypeError("Slice assignment requires a Matrix value.")
		if value.dim != (len(row_idx), len(col_idx)):
			raise ValueError("Assigned matrix shape does not match target slice shape.")

		for i, rr in enumerate(row_idx):
			for j, cc in enumerate(col_idx):
				self.data[rr][cc] = self._cast_value(value.data[i][j])

	def __len__(self):
		return self.dim[0] * self.dim[1]

	def __str__(self):
		r, c = self.dim
		if r == 0:
			return "[]"
		if c == 0:
			return "\n".join(["[]" for _ in range(r)])
		lines = []
		for row in self.data:
			lines.append("[" + " ".join(f"{float(x):>10.4f}" for x in row) + "]")
		return "\n".join(lines)

	def num_product(self, n):
		if not _is_number(n):
			raise TypeError("Scalar multiplier must be numeric.")
		out = [[val * n for val in row] for row in self.data]
		out_dtype = self.dtype if self.dtype == "fraction" and not isinstance(n, float) else "float"
		return Matrix(out, tolerance=self.tolerance, dtype=out_dtype)

	def __rmul__(self, other):
		if _is_number(other):
			return self.num_product(other)
		return NotImplemented

	def __mul__(self, other):
		if _is_number(other):
			return self.num_product(other)

		if not isinstance(other, Matrix):
			raise TypeError("Matrix multiplication requires a Matrix or a scalar.")
		if self.dim[1] != other.dim[0]:
			raise ValueError("Incompatible shapes for matrix multiplication.")

		m, k = self.dim
		_, n = other.dim
		out = []
		for i in range(m):
			row = []
			for j in range(n):
				val = 0
				for t in range(k):
					val += self.data[i][t] * other.data[t][j]
				row.append(val)
			out.append(row)

		out_dtype = "fraction" if self.dtype == other.dtype == "fraction" else "float"
		out_tol = max(self.tolerance, other.tolerance)
		return Matrix(out, tolerance=out_tol, dtype=out_dtype)

	def __matmul__(self, other):
		return self.__mul__(other)

	def __add__(self, other):
		if not isinstance(other, Matrix):
			raise TypeError("Addition requires a Matrix.")
		if self.dim != other.dim:
			raise ValueError("Shapes must match for addition.")
		out = [
			[self.data[i][j] + other.data[i][j] for j in range(self.dim[1])]
			for i in range(self.dim[0])
		]
		out_dtype = "fraction" if self.dtype == other.dtype == "fraction" else "float"
		out_tol = max(self.tolerance, other.tolerance)
		return Matrix(out, tolerance=out_tol, dtype=out_dtype)

	def __sub__(self, other):
		if not isinstance(other, Matrix):
			raise TypeError("Subtraction requires a Matrix.")
		if self.dim != other.dim:
			raise ValueError("Shapes must match for subtraction.")
		out = [
			[self.data[i][j] - other.data[i][j] for j in range(self.dim[1])]
			for i in range(self.dim[0])
		]
		out_dtype = "fraction" if self.dtype == other.dtype == "fraction" else "float"
		out_tol = max(self.tolerance, other.tolerance)
		return Matrix(out, tolerance=out_tol, dtype=out_dtype)

	def __pow__(self, n):
		if not isinstance(n, int):
			raise TypeError("Exponent must be an integer.")
		if n < 0:
			raise ValueError("Negative powers are not supported directly; use inverse() first.")
		if self.dim[0] != self.dim[1]:
			raise ValueError("Only square matrices can be exponentiated.")

		size = self.dim[0]
		result = Matrix.identity(size, tolerance=self.tolerance, dtype=self.dtype)
		base = self.copy()
		exp = n
		while exp > 0:
			if exp % 2 == 1:
				result = result * base
			base = base * base
			exp //= 2
		return result

	def gauss_elimination(self, pivoting=True):
		"""
		Row-echelon reduction by Gaussian elimination.
		Returns: (row_echelon_matrix, swap_count)

		Better method note:
		- pivoting=True is generally better numerically than pivoting=False.
		"""
		mat = copy.deepcopy(self.data)
		rows, cols = self.dim
		swap_count = 0
		tol = self.tolerance
		zero = Fraction(0) if self.dtype == "fraction" else 0.0

		r = 0
		for c in range(cols):
			if r >= rows:
				break

			pivot_row = None
			if pivoting:
				best = 0.0
				for i in range(r, rows):
					if not self._near_zero(mat[i][c], tol):
						score = abs(float(mat[i][c]))
						if score > best:
							best = score
							pivot_row = i
			else:
				for i in range(r, rows):
					if not self._near_zero(mat[i][c], tol):
						pivot_row = i
						break

			if pivot_row is None:
				continue

			if pivot_row != r:
				mat[r], mat[pivot_row] = mat[pivot_row], mat[r]
				swap_count += 1

			pivot = mat[r][c]
			for i in range(r + 1, rows):
				if self._near_zero(mat[i][c], tol):
					continue
				factor = mat[i][c] / pivot
				for j in range(c, cols):
					mat[i][j] -= factor * mat[r][j]
					if self._near_zero(mat[i][j], tol):
						mat[i][j] = zero
			r += 1

		return Matrix(mat, tolerance=self.tolerance, dtype=self.dtype), swap_count

	def determinant(self):
		if self.dim[0] != self.dim[1]:
			raise ValueError("Determinant is defined only for square matrices.")
		ref, swaps = self.gauss_elimination(pivoting=True)
		n = self.dim[0]
		det = Fraction(1) if self.dtype == "fraction" else 1.0
		for i in range(n):
			det *= ref.data[i][i]
		if swaps % 2 == 1:
			det = -det
		if self._near_zero(det, self.tolerance):
			return Fraction(0) if self.dtype == "fraction" else 0.0
		return det

	def det(self):
		return self.determinant()

	def inverse(self):
		"""
		Gauss-Jordan inverse with partial pivoting.
		Better than a no-pivot variant for numerical stability.
		"""
		n, m = self.dim
		if n != m:
			raise ValueError("Inverse is defined only for square matrices.")

		tol = self.tolerance
		zero = Fraction(0) if self.dtype == "fraction" else 0.0
		one = Fraction(1) if self.dtype == "fraction" else 1.0

		aug = [
			copy.deepcopy(self.data[i]) + [one if i == j else zero for j in range(n)]
			for i in range(n)
		]

		for col in range(n):
			pivot_row = col
			best = abs(float(aug[col][col]))
			for r in range(col + 1, n):
				score = abs(float(aug[r][col]))
				if score > best:
					best = score
					pivot_row = r

			if self._near_zero(aug[pivot_row][col], tol):
				raise ValueError("Matrix is singular and cannot be inverted.")

			if pivot_row != col:
				aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

			pivot = aug[col][col]
			for j in range(col, 2 * n):
				aug[col][j] /= pivot

			for r in range(n):
				if r == col:
					continue
				factor = aug[r][col]
				if self._near_zero(factor, tol):
					continue
				for j in range(col, 2 * n):
					aug[r][j] -= factor * aug[col][j]
					if self._near_zero(aug[r][j], tol):
						aug[r][j] = zero

		inv = [[aug[i][j + n] for j in range(n)] for i in range(n)]
		return Matrix(inv, tolerance=self.tolerance, dtype=self.dtype)

	def rank(self):
		ref, _ = self.gauss_elimination(pivoting=True)
		rank_val = 0
		for i in range(ref.dim[0]):
			if any(not self._near_zero(ref.data[i][j], self.tolerance) for j in range(ref.dim[1])):
				rank_val += 1
		return rank_val

	def lu_decomposition(self, pivoting=True):
		"""
		Doolittle LU decomposition.

		Returns:
		- pivoting=True: (P, L, U)
		- pivoting=False: (L, U)

		Better method note:
		- pivoting=True is usually better and more robust.
		"""
		n, m = self.dim
		if n != m:
			raise ValueError("LU decomposition requires a square matrix.")

		tol = self.tolerance
		zero = Fraction(0) if self.dtype == "fraction" else 0.0
		one = Fraction(1) if self.dtype == "fraction" else 1.0

		U = copy.deepcopy(self.data)
		L = [[zero for _ in range(n)] for _ in range(n)]
		P = [[one if i == j else zero for j in range(n)] for i in range(n)]

		for i in range(n):
			L[i][i] = one

		for k in range(n):
			pivot_row = k
			if pivoting:
				best = abs(float(U[k][k]))
				for r in range(k + 1, n):
					score = abs(float(U[r][k]))
					if score > best:
						best = score
						pivot_row = r

			if self._near_zero(U[pivot_row][k], tol):
				raise ValueError("Matrix is singular or LU requires pivoting at a zero pivot.")

			if pivot_row != k:
				U[k], U[pivot_row] = U[pivot_row], U[k]
				P[k], P[pivot_row] = P[pivot_row], P[k]
				for c in range(k):
					L[k][c], L[pivot_row][c] = L[pivot_row][c], L[k][c]

			for r in range(k + 1, n):
				factor = U[r][k] / U[k][k]
				L[r][k] = factor
				for c in range(k, n):
					U[r][c] -= factor * U[k][c]
					if self._near_zero(U[r][c], tol):
						U[r][c] = zero

		L_mat = Matrix(L, tolerance=self.tolerance, dtype=self.dtype)
		U_mat = Matrix(U, tolerance=self.tolerance, dtype=self.dtype)
		if pivoting:
			P_mat = Matrix(P, tolerance=self.tolerance, dtype=self.dtype)
			return P_mat, L_mat, U_mat
		return L_mat, U_mat

	def cholesky_decomposition(self):
		"""
		Cholesky decomposition A = L * L^T for SPD matrices.
		"""
		n, m = self.dim
		if n != m:
			raise ValueError("Cholesky decomposition requires a square matrix.")
		if not self._is_symmetric():
			raise ValueError("Cholesky decomposition requires a symmetric matrix.")

		A = self.to_float_data()
		tol = self.tolerance
		L = [[0.0 for _ in range(n)] for _ in range(n)]

		for i in range(n):
			for j in range(i + 1):
				s = sum(L[i][k] * L[j][k] for k in range(j))
				if i == j:
					val = A[i][i] - s
					if val <= tol:
						raise ValueError("Matrix is not positive definite.")
					L[i][j] = math.sqrt(val)
				else:
					if abs(L[j][j]) <= tol:
						raise ValueError("Matrix is not positive definite.")
					L[i][j] = (A[i][j] - s) / L[j][j]

		L_mat = Matrix(L, tolerance=self.tolerance, dtype="float")
		return L_mat, L_mat.transpose()

	def gram_schmidt(self, method="mgs"):
		"""
		Orthonormalize columns of A.

		method:
		- 'cgs' / 'classical': Classical Gram-Schmidt (less stable)
		- 'mgs' / 'modified': Modified Gram-Schmidt (better)
		"""
		m, n = self.dim
		if m == 0 or n == 0:
			raise ValueError("Cannot run Gram-Schmidt on an empty matrix.")

		method_key = method.lower()
		if method_key not in {"cgs", "classical", "mgs", "modified"}:
			raise ValueError("method must be 'cgs'/'classical' or 'mgs'/'modified'.")

		cols = [[float(self.data[i][j]) for i in range(m)] for j in range(n)]
		q_cols = []
		tol = self.tolerance

		if method_key in {"cgs", "classical"}:
			for col in cols:
				v = col[:]
				for q in q_cols:
					proj = self._dot(col, q)
					v = [v[i] - proj * q[i] for i in range(m)]
				norm_v = self._norm(v)
				if norm_v <= tol:
					raise ValueError("Columns are linearly dependent; CGS failed.")
				q_cols.append([x / norm_v for x in v])
		else:
			for col in cols:
				v = col[:]
				for q in q_cols:
					proj = self._dot(v, q)
					v = [v[i] - proj * q[i] for i in range(m)]
				norm_v = self._norm(v)
				if norm_v <= tol:
					raise ValueError("Columns are linearly dependent; MGS failed.")
				q_cols.append([x / norm_v for x in v])

		q_data = [[q_cols[j][i] for j in range(n)] for i in range(m)]
		return Matrix(q_data, tolerance=self.tolerance, dtype="float")

	def _qr_householder(self):
		m, n = self.dim
		if m == 0 or n == 0:
			raise ValueError("Cannot run QR on an empty matrix.")

		R = self.to_float_data()
		Q = Matrix.identity(m, tolerance=self.tolerance, dtype="float").to_float_data()
		k_max = min(m, n)
		tol = self.tolerance

		for k in range(k_max):
			x = [R[i][k] for i in range(k, m)]
			norm_x = self._norm(x)
			if norm_x <= tol:
				continue

			sign = -1.0 if x[0] < 0 else 1.0
			v = x[:]
			v[0] += sign * norm_x
			norm_v = self._norm(v)
			if norm_v <= tol:
				continue
			v = [val / norm_v for val in v]

			for j in range(k, n):
				dot_vr = sum(v[i] * R[k + i][j] for i in range(len(v)))
				for i in range(len(v)):
					R[k + i][j] -= 2.0 * v[i] * dot_vr

			for i in range(m):
				dot_qv = sum(v[t] * Q[i][k + t] for t in range(len(v)))
				for t in range(len(v)):
					Q[i][k + t] -= 2.0 * v[t] * dot_qv

		for i in range(m):
			for j in range(min(i, n)):
				if abs(R[i][j]) <= tol:
					R[i][j] = 0.0

		return (
			Matrix(Q, tolerance=self.tolerance, dtype="float"),
			Matrix(R, tolerance=self.tolerance, dtype="float"),
		)

	def qr_decomposition(self, method="householder"):
		"""
		QR decomposition A = Q * R.

		method:
		- 'householder' (recommended)
		- 'mgs' / 'modified'
		- 'cgs' / 'classical'
		"""
		method_key = method.lower()

		if method_key in {"householder", "hh"}:
			return self._qr_householder()

		if method_key in {"mgs", "modified", "cgs", "classical"}:
			Q = self.gram_schmidt(method=method_key)
			R = Q.transpose() * Matrix(self.to_float_data(), tolerance=self.tolerance, dtype="float")

			r_data = R.to_float_data()
			rows_r, cols_r = R.dim
			for i in range(rows_r):
				for j in range(cols_r):
					if i > j and abs(r_data[i][j]) <= self.tolerance:
						r_data[i][j] = 0.0
			return Q, Matrix(r_data, tolerance=self.tolerance, dtype="float")

		raise ValueError("Unknown QR method.")

	def QR_decomposition(self, method="householder"):
		return self.qr_decomposition(method=method)

	def eigenvalues_qr(self, iterations=100, qr_method="householder"):
		"""
		Approximate eigenvalues via unshifted QR iteration.
		Better for general matrices than Jacobi (which is symmetric-only).
		"""
		n, m = self.dim
		if n != m:
			raise ValueError("Eigenvalues are defined only for square matrices.")

		current = Matrix(self.to_float_data(), tolerance=self.tolerance, dtype="float")
		for _ in range(iterations):
			Q, R = current.qr_decomposition(method=qr_method)
			current = R * Q

		vals = [float(current.data[i][i]) for i in range(n)]
		return [0.0 if abs(v) <= self.tolerance else v for v in vals]

	def eigenpairs_jacobi(self, max_iterations=1000, sort=True):
		"""
		Jacobi eigenvalue algorithm for real symmetric matrices.
		Returns: (eigenvalues, eigenvector_matrix_with_columns_as_vectors)
		"""
		n, m = self.dim
		if n != m:
			raise ValueError("Eigenvalues are defined only for square matrices.")
		if not self._is_symmetric():
			raise ValueError("Jacobi method requires a symmetric matrix.")

		A = self.to_float_data()
		V = Matrix.identity(n, tolerance=self.tolerance, dtype="float").to_float_data()
		tol = self.tolerance

		for _ in range(max_iterations):
			max_val = 0.0
			p = -1
			q = -1
			for i in range(n):
				for j in range(i + 1, n):
					candidate = abs(A[i][j])
					if candidate > max_val:
						max_val = candidate
						p = i
						q = j

			if max_val < tol:
				break

			tau = (A[q][q] - A[p][p]) / (2.0 * A[p][q])
			sign_tau = 1.0 if tau >= 0 else -1.0
			t = sign_tau / (abs(tau) + math.sqrt(tau * tau + 1.0))
			c = 1.0 / math.sqrt(1.0 + t * t)
			s = t * c

			for i in range(n):
				if i != p and i != q:
					aip = A[i][p]
					aiq = A[i][q]
					A[i][p] = c * aip - s * aiq
					A[p][i] = A[i][p]
					A[i][q] = c * aiq + s * aip
					A[q][i] = A[i][q]

			app = A[p][p]
			aqq = A[q][q]
			apq = A[p][q]
			A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
			A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
			A[p][q] = 0.0
			A[q][p] = 0.0

			for i in range(n):
				vip = V[i][p]
				viq = V[i][q]
				V[i][p] = c * vip - s * viq
				V[i][q] = c * viq + s * vip

		eigenvalues = [A[i][i] if abs(A[i][i]) > tol else 0.0 for i in range(n)]

		if sort:
			idx = sorted(range(n), key=lambda i: eigenvalues[i], reverse=True)
			eigenvalues = [eigenvalues[i] for i in idx]
			V = [[V[r][i] for i in idx] for r in range(n)]

		return eigenvalues, Matrix(V, tolerance=self.tolerance, dtype="float")

	def eigenpairs(self, method="auto", iterations=100, max_iterations=1000, qr_method="householder"):
		"""
		Unified eigen API.

		Returns:
		- (eigenvalues, eigenvectors) for Jacobi
		- (eigenvalues, None) for QR iteration
		"""
		method_key = method.lower()
		if method_key == "auto":
			method_key = "jacobi" if self._is_symmetric() else "qr"

		if method_key == "jacobi":
			return self.eigenpairs_jacobi(max_iterations=max_iterations, sort=True)

		if method_key == "qr":
			return self.eigenvalues_qr(iterations=iterations, qr_method=qr_method), None

		raise ValueError("method must be 'auto', 'jacobi', or 'qr'.")

	def eigenvalues(self, method="auto", iterations=100, max_iterations=1000, qr_method="householder"):
		vals, _ = self.eigenpairs(
			method=method,
			iterations=iterations,
			max_iterations=max_iterations,
			qr_method=qr_method,
		)
		return vals

	def singular_value_decomposition(self):
		"""
		Full SVD: A = U * Sigma * V^T

		Strategy:
		- Compute eigenpairs of A^T A to get V and singular values.
		- Build U via u_i = A v_i / sigma_i (better matching than independently diagonalizing A A^T).
		- Complete U to an orthonormal basis when needed.
		"""
		m, n = self.dim
		tol = self.tolerance

		A_float = self.to_float_data()
		AtA = Matrix((self.transpose() * self).to_float_data(), tolerance=tol, dtype="float")
		eigvals, V = AtA.eigenpairs_jacobi(max_iterations=2000, sort=True)

		eigvals = [ev if ev > tol else 0.0 for ev in eigvals]
		singular_values = [math.sqrt(ev) for ev in eigvals]

		k = min(m, n)
		sigma_data = [[0.0 for _ in range(n)] for _ in range(m)]
		for i in range(k):
			sigma_data[i][i] = singular_values[i]

		V_data = V.to_float_data()
		v_cols = [[V_data[r][c] for r in range(n)] for c in range(n)]

		u_cols = [None for _ in range(k)]
		for i in range(k):
			sigma = singular_values[i]
			if sigma <= tol:
				continue
			Av = self._mat_vec_mul(A_float, v_cols[i])
			u_cols[i] = self._normalize_vec([x / sigma for x in Av], tol)

		known_u = [col for col in u_cols if col is not None]
		full_basis = self._complete_orthonormal_basis(known_u, m, tol)

		basis_cursor = len(known_u)
		for i in range(k):
			if u_cols[i] is None:
				u_cols[i] = full_basis[basis_cursor]
				basis_cursor += 1

		while len(u_cols) < m:
			u_cols.append(full_basis[basis_cursor])
			basis_cursor += 1

		U_data = [[u_cols[col][row] for col in range(m)] for row in range(m)]
		U = Matrix(U_data, tolerance=tol, dtype="float")
		Sigma = Matrix(sigma_data, tolerance=tol, dtype="float")
		V_T = Matrix(V_data, tolerance=tol, dtype="float").transpose()
		return U, Sigma, V_T

	def svd(self):
		return self.singular_value_decomposition()

	def jordan_normal_form(self):
		"""
		Jordan normal form for matrices with rational eigenvalues.
		The implementation follows Faddeev-LeVerrier + rational root theorem.
		"""
		n, m = self.dim
		if n != m:
			raise ValueError("Jordan normal form is defined only for square matrices.")
		if n == 0:
			return Matrix(data=[], tolerance=self.tolerance, dtype=self.dtype)

		A_mat = Matrix(
			data=[[Fraction(x) for x in row] for row in self.data],
			tolerance=self.tolerance,
			dtype="fraction",
		)

		c = [Fraction(0)] * (n + 1)
		c[n] = Fraction(1)

		M_mat = Matrix([[Fraction(0)] * n for _ in range(n)], tolerance=self.tolerance, dtype="fraction")
		for k in range(1, n + 1):
			c_prev = c[n - k + 1]
			c_prev_I = Matrix(
				[[c_prev if i == j else Fraction(0) for j in range(n)] for i in range(n)],
				tolerance=self.tolerance,
				dtype="fraction",
			)
			M_mat = (A_mat * M_mat) + c_prev_I
			AM = A_mat * M_mat
			trace_AM = sum(AM.data[i][i] for i in range(n))
			c[n - k] = -Fraction(1, k) * trace_AM

		def get_factors(num):
			num = abs(int(num))
			if num == 0:
				return [1]
			fac = set()
			for i in range(1, int(math.sqrt(num)) + 2):
				if num % i == 0:
					fac.add(i)
					fac.add(num // i)
			return list(fac)

		def evaluate_poly(poly, x):
			res = Fraction(0)
			for coeff in reversed(poly):
				res = res * x + coeff
			return res

		def synthetic_div(poly, root):
			new_poly = []
			val = Fraction(0)
			for coeff in reversed(poly):
				val = val * root + coeff
				new_poly.append(val)
			new_poly.pop()
			return new_poly[::-1]

		roots = {}
		poly = [coeff for coeff in c]

		while len(poly) > 1 and poly[0] == 0:
			roots[Fraction(0)] = roots.get(Fraction(0), 0) + 1
			poly = poly[1:]

		if len(poly) > 1:
			def lcm(a, b):
				return abs(a * b) // math.gcd(a, b)

			lcm_den = 1
			for coeff in poly:
				lcm_den = lcm(lcm_den, coeff.denominator)

			int_coeffs = [int(coeff * lcm_den) for coeff in poly]
			c0 = int_coeffs[0]
			cn = int_coeffs[-1]

			p_candidates = get_factors(c0)
			q_candidates = get_factors(cn)
			candidates = set()
			for p in p_candidates:
				for q in q_candidates:
					candidates.add(Fraction(p, q))
					candidates.add(Fraction(-p, q))

			for cand in list(candidates):
				while len(int_coeffs) > 1 and evaluate_poly(int_coeffs, cand) == 0:
					roots[cand] = roots.get(cand, 0) + 1
					int_coeffs = synthetic_div(int_coeffs, cand)

			if len(int_coeffs) > 1:
				raise ValueError("Matrix has non-rational eigenvalues; exact Jordan form is unsupported here.")

		jordan_blocks = []
		for eigenvalue, alg_mult in roots.items():
			diag_B = Matrix(
				[[-eigenvalue if i == j else Fraction(0) for j in range(n)] for i in range(n)],
				tolerance=self.tolerance,
				dtype="fraction",
			)
			B = A_mat + diag_B

			ranks = [n]
			current_B = Matrix.identity(n, tolerance=self.tolerance, dtype="fraction")

			k_max = alg_mult
			for k in range(1, k_max + 2):
				current_B = current_B * B
				ranks.append(current_B.rank())
				if len(ranks) >= 3 and ranks[-1] == ranks[-2]:
					while len(ranks) <= k_max + 2:
						ranks.append(ranks[-1])
					break

			for k in range(1, alg_mult + 1):
				if k + 1 < len(ranks):
					num_blocks = ranks[k - 1] - 2 * ranks[k] + ranks[k + 1]
					for _ in range(num_blocks):
						jordan_blocks.append((eigenvalue, k))

		J_data = [[Fraction(0)] * n for _ in range(n)]
		idx = 0
		for eigenvalue, block_size in jordan_blocks:
			for i in range(block_size):
				J_data[idx + i][idx + i] = eigenvalue
				if i < block_size - 1:
					J_data[idx + i][idx + i + 1] = Fraction(1)
			idx += block_size

		return Matrix(J_data, tolerance=self.tolerance, dtype="fraction")


def I(n, tolerance=1e-10, dtype="fraction"):
	return Matrix.identity(n, tolerance=tolerance, dtype=dtype)


def concatenate(items, axis=0):
	items = list(items)
	if len(items) == 0:
		raise ValueError("No matrices to concatenate.")
	for item in items:
		if not isinstance(item, Matrix):
			raise TypeError("All items must be Matrix objects.")

	result = items[0].copy()
	for item in items[1:]:
		result = result.concatenate(item, index=axis)
	return result
