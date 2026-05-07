import random
import copy
from fractions import Fraction

class Matrix:
	def __init__(self, data=None, dim=None, init_value=0):
		"""
		Initialize the Matrix. 
		Matches the error reporting style of Code 2.
		"""
		if data == None and dim == None:
			raise ValueError("1-1: Lack enough variables")
		if data is not None:
			if not isinstance(data, list):
				raise TypeError("1-2: The data should be a nested list")
			if len(data) > 0:
				for i in range(len(data)):
					if not isinstance(data[i], list) or len(data[i]) != len(data[0]):
						raise TypeError("1-4: Data must be a consistent nested list")
				self.dim = (len(data), len(data[0]))
			else:
				self.dim = (0, 0)
			self.data = data
		else:
			if not isinstance(dim, tuple) or len(dim) != 2:
				raise ValueError("1-6: Dim must be a tuple of (rows, cols)")
			self.data = [[init_value for _ in range(dim[1])] for _ in range(dim[0])]
			self.dim = dim
		self.init_value = init_value

	def shape(self):
		"""Return the shape of class matrix"""
		return self.dim

	def reshape(self, newdim):
		"""Reshape the matrix without changing its data"""
		m, n = self.dim
		s, t = newdim
		if m * n != s * t:
			raise ValueError("3-4: You can't change the total size of the matrix")
		
		flat = [item for row in self.data for item in row]
		res = []
		for i in range(s):
			res.append(flat[i*t : (i+1)*t])
		return Matrix(data=res)

	def dot(self, other):
		"""Matrix multiplication (dot product)"""
		if self.dim[1] != other.dim[0]:
			raise TypeError("4-3: These two matrix can't be multiplied")
		
		res = [[0 for _ in range(other.dim[1])] for _ in range(self.dim[0])]
		for i in range(self.dim[0]):
			for k in range(self.dim[1]):
				for j in range(other.dim[1]):
					res[i][j] += self.data[i][k] * other.data[k][j]
		return Matrix(data=res)

	def T(self):
		"""Transpose the matrix"""
		res = [[self.data[j][i] for j in range(self.dim[0])] for i in range(self.dim[1])]
		return Matrix(data=res)

	def sum(self, axis=None):
		"""Sum the matrix by axis"""
		if axis == None:
			total = sum(sum(row) for row in self.data)
			return Matrix(data=[[total]])
		if axis == 0:
			res = [[sum(row[i] for row in self.data) for i in range(self.dim[1])]]
			return Matrix(data=res)
		if axis == 1:
			res = [[sum(row)] for row in self.data]
			return Matrix(data=res)

	def copy(self):
		"""Deepcopy the matrix"""
		return Matrix(data=copy.deepcopy(self.data))

	def Kronecker_product(self, other):
		"""Calculate the Kronecker product"""
		m1, n1 = self.dim
		m2, n2 = other.dim
		res = [[0] * (n1 * n2) for _ in range(m1 * m2)]
		for i in range(m1):
			for j in range(n1):
				for p in range(m2):
					for q in range(n2):
						res[i * m2 + p][j * n2 + q] = self.data[i][j] * other.data[p][q]
		return Matrix(data=res)

	def __getitem__(self, key):
		"""Get elements or sub-matrix using slicing"""
		row_key, col_key = key
		def get_indices(k, limit):
			if isinstance(k, int): return [k if k >= 0 else limit + k]
			return range(*k.indices(limit))
		
		rows = get_indices(row_key, self.dim[0])
		cols = get_indices(col_key, self.dim[1])
		
		if isinstance(row_key, int) and isinstance(col_key, int):
			return self.data[rows[0]][cols[0]]
		
		res = [[self.data[r][c] for c in cols] for r in rows]
		return Matrix(data=res)

	def __setitem__(self, key, value):
		"""Set elements or sub-matrix values"""
		row_key, col_key = key
		def get_indices(k, limit):
			if isinstance(k, int): return [k if k >= 0 else limit + k]
			return range(*k.indices(limit))
		
		rows = get_indices(row_key, self.dim[0])
		cols = get_indices(col_key, self.dim[1])
		
		if isinstance(row_key, int) and isinstance(col_key, int):
			self.data[rows[0]][cols[0]] = value
		else:
			val_data = value.data if isinstance(value, Matrix) else value
			for i, r in enumerate(rows):
				for j, c in enumerate(cols):
					self.data[r][c] = val_data[i][j]

	def __pow__(self, n):
		"""Matrix exponentiation using binary exponentiation (Fast Power)"""
		if self.dim[0] != self.dim[1]:
			raise ValueError("11-4: Only square matrix can be exponentiated")
		if n == 0: return I(self.dim[0])
		
		res = I(self.dim[0])
		base = self.copy()
		while n > 0:
			if n % 2 == 1: res = res.dot(base)
			base = base.dot(base)
			n //= 2
		return res

	def __add__(self, other):
		"""Matrix addition"""
		res = [[self.data[i][j] + other.data[i][j] for j in range(self.dim[1])] for i in range(self.dim[0])]
		return Matrix(data=res)

	def __sub__(self, other):
		"""Matrix subtraction"""
		res = [[self.data[i][j] - other.data[i][j] for j in range(self.dim[1])] for i in range(self.dim[0])]
		return Matrix(data=res)

	def __mul__(self, other):
		"""Hadamard product (element-wise multiplication)"""
		res = [[self.data[i][j] * other.data[i][j] for j in range(self.dim[1])] for i in range(self.dim[0])]
		return Matrix(data=res)

	def __len__(self):
		"""Return total number of elements"""
		return self.dim[0] * self.dim[1]

	def __str__(self):
		"""Formatted string representation"""
		max_l = max(len(str(x)) for row in self.data for x in row)
		lines = ["[" + " ".join(f"{x:>{max_l}}" for x in row) + "]" for row in self.data]
		return "[" + "\n ".join(lines) + "]"

	def det(self):
		"""Calculate determinant using Gaussian elimination with Fractions"""
		if self.dim[0] != self.dim[1]: raise ValueError("Square matrix required")
		n = self.dim[0]
		mat = [[Fraction(x) for x in row] for row in self.data]
		sign = 1
		for i in range(n):
			pivot = i
			while pivot < n and mat[pivot][i] == 0: pivot += 1
			if pivot == n: return 0
			if pivot != i:
				mat[i], mat[pivot] = mat[pivot], mat[i]
				sign *= -1
			for j in range(i + 1, n):
				mult = mat[j][i] / mat[i][i]
				for k in range(i, n): mat[j][k] -= mult * mat[i][k]
		res = sign
		for i in range(n): res *= mat[i][i]
		return float(res) if res.denominator != 1 else int(res)

	def inverse(self):
		"""Calculate inverse matrix using augmented matrix [A|I]"""
		if self.dim[0] != self.dim[1]: raise ValueError("18-2: Only square matrices have inverses")
		n = self.dim[0]
		aug = [[Fraction(self.data[i][j]) for j in range(n)] + [Fraction(1 if i == k else 0) for k in range(n)] for i in range(n)]
		for i in range(n):
			pivot = i
			while pivot < n and aug[pivot][i] == 0: pivot += 1
			if pivot == n: raise ValueError("18-3: Matrix is singular")
			aug[i], aug[pivot] = aug[pivot], aug[i]
			factor = aug[i][i]
			for j in range(i, 2 * n): aug[i][j] /= factor
			for j in range(n):
				if i != j:
					mult = aug[j][i]
					for k in range(i, 2 * n): aug[j][k] -= mult * aug[i][k]
		inv = [[float(aug[i][j]) if aug[i][j].denominator != 1 else int(aug[i][j]) for j in range(n, 2 * n)] for i in range(n)]
		return Matrix(data=inv)

	def rank(self):
		"""Calculate the rank of the matrix"""
		n, m = self.dim
		mat = [[Fraction(x) for x in row] for row in self.data]
		r = 0
		for i in range(m):
			if r >= n: break
			pivot = r
			while pivot < n and mat[pivot][i] == 0: pivot += 1
			if pivot == n: continue
			mat[r], mat[pivot] = mat[pivot], mat[r]
			for j in range(r + 1, n):
				mult = mat[j][i] / mat[r][i]
				for k in range(i, m): mat[j][k] -= mult * mat[r][k]
			r += 1
		return r

def I(n):
	"""Return an n*n unit matrix"""
	return Matrix(data=[[1 if i == j else 0 for j in range(n)] for i in range(n)])

def narray(dim, init_value=1):
	return Matrix(dim=dim, init_value=init_value)

def arange(start, end, step):
	data = []
	curr = start
	while (step > 0 and curr < end) or (step < 0 and curr > end):
		data.append(curr)
		curr += step
	return Matrix(data=[data])

def zeros(dim):
	return Matrix(dim=dim, init_value=0)

def zeros_like(matrix):
	return zeros(matrix.dim)

def ones(dim):
	return Matrix(dim=dim, init_value=1)

def ones_like(matrix):
	return ones(matrix.dim)

def nrandom(dim):
	res = [[random.random() for _ in range(dim[1])] for _ in range(dim[0])]
	return Matrix(data=res)

def nrandom_like(matrix):
	return nrandom(matrix.dim)

def concatenate(items, axis=0):
	items = list(items)
	if axis == 0:
		res_data = []
		for it in items: res_data.extend(it.data)
	else:
		res_data = [[] for _ in range(items[0].dim[0])]
		for it in items:
			for i in range(it.dim[0]): res_data[i].extend(it.data[i])
	return Matrix(data=res_data)

def vectorize(func):
	def f(x):
		res_data = [[func(v) for v in row] for row in x.data]
		return Matrix(data=res_data)
	return f