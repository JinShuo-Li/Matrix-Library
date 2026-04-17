# this is a simple implementation of several matrix operations

from copy import deepcopy
import math
import random

# helper function

def random_matrix(rows, cols, left=0, right=10):
    return [[random.randint(left, right) for _ in range(cols)] for _ in range(rows)]

# matrix class

class Matrix:
    def __init__(self, data):
        self.data = [[float(x) for x in row] for row in data]
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0
    
    def __str__(self):
        return '\n'.join(['\t'.join(f"{value:.3f}" for value in row) for row in self.data])
    
    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for addition")
        result = [[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result)
    
    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for subtraction")
        result = [[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result)
    
    def __mul__(self, other):
        if self.cols != other.rows:
            raise ValueError("Number of columns of the first matrix must equal the number of rows of the second matrix for multiplication")
        result = [[sum(self.data[i][k] * other.data[k][j] for k in range(self.cols)) for j in range(other.cols)] for i in range(self.rows)]
        return Matrix(result)
    
    def transpose(self):
        result = [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]
        return Matrix(result)
    
    def num_product(self, scalar):
        result = [[self.data[i][j] * scalar for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result)

    @staticmethod
    def _from_float_matrix(data, eps=1e-12):
        cleaned = []
        for row in data:
            cleaned_row = []
            for value in row:
                if abs(value) < eps:
                    value = 0.0
                cleaned_row.append(float(value))
            cleaned.append(cleaned_row)
        return Matrix(cleaned)
    
    def gaussian_elimination(self):
        mat = deepcopy(self.data)
        r = 0

        for c in range(self.cols):
            if r >= self.rows:
                break

            max_row = max(range(r, self.rows), key=lambda i: abs(mat[i][c]))

            if abs(mat[max_row][c]) < 1e-12:
                continue

            mat[r], mat[max_row] = mat[max_row], mat[r]

            for j in range(r + 1, self.rows):
                if abs(mat[j][c]) >= 1e-12:
                    factor = mat[j][c] / mat[r][c]
                    for k in range(c, self.cols):
                        mat[j][k] -= factor * mat[r][k]

            r += 1
        return Matrix(mat)
    
    def qr_decomposition_housholder(self):
        if self.rows == 0 or self.cols == 0:
            raise ValueError("Matrix must be non-empty for QR decomposition")

        m, n = self.rows, self.cols
        r = [[float(self.data[i][j]) for j in range(n)] for i in range(m)]
        q_t = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]

        for c in range(min(m, n)):
            x = [r[i][c] for i in range(c, m)]
            norm_x = math.sqrt(sum(value * value for value in x))

            if norm_x == 0.0:
                continue

            sign = 1.0 if x[0] >= 0.0 else -1.0
            v = x[:]
            v[0] += sign * norm_x
            norm_v = math.sqrt(sum(value * value for value in v))

            if norm_v == 0.0:
                continue

            v = [value / norm_v for value in v]

            # Apply Householder reflection to R.
            for j in range(c, n):
                dot = sum(v[i] * r[c + i][j] for i in range(len(v)))
                for i in range(len(v)):
                    r[c + i][j] -= 2.0 * v[i] * dot

            # Apply Householder reflection to Q^T.
            for j in range(m):
                dot = sum(v[i] * q_t[c + i][j] for i in range(len(v)))
                for i in range(len(v)):
                    q_t[c + i][j] -= 2.0 * v[i] * dot

        q = [[q_t[j][i] for j in range(m)] for i in range(m)]
        return Matrix._from_float_matrix(q), Matrix._from_float_matrix(r)

    def qr_decomposition_givens(self):
        if self.rows == 0 or self.cols == 0:
            raise ValueError("Matrix must be non-empty for QR decomposition")

        m, n = self.rows, self.cols
        r = [[float(self.data[i][j]) for j in range(n)] for i in range(m)]
        q_t = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]

        for c in range(min(m, n)):
            for i in range(m - 1, c, -1):
                a = r[c][c]
                b = r[i][c]

                if abs(b) < 1e-15:
                    continue

                radius = math.hypot(a, b)
                cos_theta = a / radius
                sin_theta = -b / radius

                # Left-multiply R by a Givens rotation in rows c and i.
                for j in range(c, n):
                    top = cos_theta * r[c][j] - sin_theta * r[i][j]
                    bottom = sin_theta * r[c][j] + cos_theta * r[i][j]
                    r[c][j], r[i][j] = top, bottom

                # Accumulate the same rotations into Q^T.
                for j in range(m):
                    top = cos_theta * q_t[c][j] - sin_theta * q_t[i][j]
                    bottom = sin_theta * q_t[c][j] + cos_theta * q_t[i][j]
                    q_t[c][j], q_t[i][j] = top, bottom

        q = [[q_t[j][i] for j in range(m)] for i in range(m)]
        return Matrix._from_float_matrix(q), Matrix._from_float_matrix(r)


if __name__ == "__main__":
    A = Matrix([
        [3, 14,  9],
        [6, 43,  3],
        [6, 22, 15]
    ])
    B = Matrix([
        [2, 2, 1],
        [0, 2, 2],
        [2, 1, 2]
    ])
    print("A:")
    print(A)
    print("\nB:")
    print(B)
    print("A's QR decomposition (Householder):")
    Q, R = A.qr_decomposition_housholder()
    print("Q:")
    print(Q)
    print("R:")
    print(R)
    print("Q * R:")
    print(Q * R)
    print("\nB's QR decomposition (Givens):")
    Q, R = B.qr_decomposition_givens()
    print("Q:")
    print(Q)
    print("R:")
    print(R)
    print("Q * R:")
    print(Q * R)