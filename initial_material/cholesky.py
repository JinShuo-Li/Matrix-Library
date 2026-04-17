from fractions import Fraction

class Matrix:
    def __init__(self, matrix):
        self.matrix = [[Fraction(x) for x in row] for row in matrix]
        self.size = (len(matrix), len(matrix[0]) if matrix else 0)

    def __add__(self, other):
        if self.size != other.size:
            raise ValueError("Matrices must be of the same size for addition.")
        
        result = [[self.matrix[i][j] + other.matrix[i][j] for j in range(self.size[1])] for i in range(self.size[0])]
        return Matrix(result)
    
    def __mul__(self, other):
        if self.size[1] != other.size[0]:
            raise ValueError("Number of columns of the first matrix must equal the number of rows of the second matrix for multiplication.")
        
        result = [[sum(self.matrix[i][k] * other.matrix[k][j] for k in range(self.size[1])) for j in range(other.size[1])] for i in range(self.size[0])]
        return Matrix(result)

    def __str__(self):
        return '\n'.join(['[' + '\t'.join(str(float(x)) for x in row) + ']' for row in self.matrix])
    
    def num_product(self, n):
        result = [[self.matrix[i][j] * n for j in range(self.size[1])] for i in range(self.size[0])]
        return Matrix(result)

    def concatenate(self, other, index):
        if index == 0:
            if self.size[1] != other.size[1]:
                raise ValueError("Matrices must have the same number of columns for row concatenation.")
            result = self.matrix + other.matrix
        elif index == 1:
            if self.size[0] != other.size[0]:
                raise ValueError("Matrices must have the same number of rows for column concatenation.")
            result = [self.matrix[i] + other.matrix[i] for i in range(self.size[0])]
        else:
            raise ValueError("Index must be 0 or 1.")
        
        return Matrix(result)

    def transpose(self):
        result = [[self.matrix[j][i] for j in range(self.size[0])] for i in range(self.size[1])]
        return Matrix(result)
    
    def gauss_elimination(self):
        new_matrix = [[x for x in row] for row in self.matrix]
        rows, cols = self.size
        swap_count = 0
        
        for i in range(rows):
            if i >= cols:
                break

            pivot_row = i
            while pivot_row < rows and new_matrix[pivot_row][i] == 0:
                pivot_row += 1
            
            if pivot_row == rows:
                continue
            
            if i != pivot_row:
                new_matrix[i], new_matrix[pivot_row] = new_matrix[pivot_row], new_matrix[i]
                swap_count += 1
            
            for j in range(i + 1, rows):
                if new_matrix[j][i] != 0:
                    factor = new_matrix[j][i] / new_matrix[i][i]
                    for k in range(i, cols):
                        new_matrix[j][k] -= factor * new_matrix[i][k]
                        
        return Matrix(new_matrix), swap_count
    
    def inverse(self):
        rows, cols = self.size
        if rows != cols:
            raise ValueError("Matrix must be square to find its inverse.")
        
        E = Matrix([[Fraction(1) if i == j else Fraction(0) for j in range(cols)] for i in range(rows)])
        M_E = self.concatenate(E, 1)
        new_matrix = [[x for x in row] for row in M_E.matrix]
        
        for i in range(rows):
            pivot_row = i
            while pivot_row < rows and new_matrix[pivot_row][i] == 0:
                pivot_row += 1
            
            if pivot_row == rows:
                raise ValueError("Matrix is singular and cannot be inverted.")
            
            if i != pivot_row:
                new_matrix[i], new_matrix[pivot_row] = new_matrix[pivot_row], new_matrix[i]
            
            pivot_val = new_matrix[i][i]
            for j in range(i, cols * 2):
                new_matrix[i][j] /= pivot_val
            
            for j in range(rows):
                if i != j:
                    factor = new_matrix[j][i]
                    for k in range(i, cols * 2):
                        new_matrix[j][k] -= factor * new_matrix[i][k]
                        
        inv_matrix = [[new_matrix[i][j + cols] for j in range(cols)] for i in range(rows)]
        return Matrix(inv_matrix)

    def determinant(self):
        if self.size[0] != self.size[1]:
            raise ValueError("Determinant is only defined for square matrices.")
        
        mat, swap_count = self.gauss_elimination()
        det = Fraction(1)
        for i in range(self.size[0]):
            det *= mat.matrix[i][i]
        
        return det * (-1)**swap_count

    def lu_decomposition(self):
        new_matrix = Matrix([[x for x in row] for row in self.matrix])
        rows, cols = self.size
        if rows != cols:
            raise ValueError("LU decomposition requires a square matrix.")
        E = Matrix([[Fraction(1) if i == j else Fraction(0) for j in range(cols)] for i in range(rows)])
        M_E = new_matrix.concatenate(E, 1)
        M_E_gauss, swap_count = M_E.gauss_elimination()
        if swap_count != 0:
            raise ValueError("Matrix cannot be decomposed into LU form.")

        U_part = [[M_E_gauss.matrix[i][j] for j in range(cols)] for i in range(rows)]

        L_inv_part = [[M_E_gauss.matrix[i][j + cols] for j in range(cols)] for i in range(rows)]
        
        return Matrix(L_inv_part).inverse(), Matrix(U_part)

    def cholesky_decomposition(self):
        if self.size[0] != self.size[1]:
            raise ValueError("Cholesky decomposition requires a square matrix.")
        
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self.matrix[i][j] != self.matrix[j][i]:
                    raise ValueError("Matrix must be symmetric for Cholesky decomposition.")
        
        n = self.size[0]
        L_mat = [[Fraction(0) for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1):
                sum_k = sum(L_mat[i][k] * L_mat[j][k] for k in range(j))
                
                if i == j:
                    val = self.matrix[i][i] - sum_k
                    if val <= 0:
                        raise ValueError("Matrix is not positive definite.")
                    L_mat[i][j] = Fraction(float(val) ** 0.5)
                else:
                    L_mat[i][j] = (self.matrix[i][j] - sum_k) / L_mat[j][j]
                    
        L = Matrix(L_mat)
        return L, L.transpose()

    def gram_schmidt(self):
        rows, cols = self.size
        q_cols = []
        for j in range(cols):
            v = [self.matrix[i][j] for i in range(rows)]
            for e in q_cols:
                dot_val = sum(v[i] * e[i] for i in range(rows))
                v = [v[i] - dot_val * e[i] for i in range(rows)]
            
            norm = sum(x * x for x in v)
            if norm == 0:
                raise ValueError("Columns are linearly dependent, cannot perform Gram-Schmidt.")
            norm_val = Fraction(float(norm) ** 0.5)
            e_new = [x / norm_val for x in v]
            q_cols.append(e_new)
            
        Q_mat = [[q_cols[j][i] for j in range(cols)] for i in range(rows)]
        return Matrix(Q_mat)

    def qr_decomposition(self):
        Q = self.gram_schmidt()
        R = Q.transpose() * self

        R_clean = [[R.matrix[i][j] if i <= j else Fraction(0) for j in range(R.size[1])] for i in range(R.size[0])]
        return Q, Matrix(R_clean)

    def eigenvalues(self, iterations=50):
        if self.size[0] != self.size[1]:
            raise ValueError("Eigenvalues are only defined for square matrices.")
        
        current_matrix = self
        for _ in range(iterations):
            try:
                Q, R = current_matrix.qr_decomposition()
                current_matrix = R * Q
            except ValueError:
                # If QR decomposition fails (e.g., due to linearly dependent columns), stop early.
                break
                
        return [current_matrix.matrix[i][i] for i in range(self.size[0])]


def main():
    print("\nTesting the QR decomposition:")
    mat2 = Matrix([
        [ 2, -4],
        [-1,  3],
        [ 2,  2]
    ])
    Q, R = mat2.qr_decomposition()
    print("Q:")
    print(Q)
    print("R:")
    print(R)
    print("Q * R:")
    print(Q * R)

if __name__ == "__main__":
    main()


