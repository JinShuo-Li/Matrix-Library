from fractions import Fraction
import random

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
        
    def __str__(self):
        return '\n'.join(['[' + '\t'.join(str(x) for x in row) + ']' for row in self.matrix])

    def determinant(self):
        if self.size[0] != self.size[1]:
            raise ValueError("Determinant is only defined for square matrices.")
        
        mat, swap_count = self.gauss_elimination()
        det = Fraction(1)
        for i in range(self.size[0]):
            det *= mat.matrix[i][i]
        
        return det * (-1)**swap_count

if __name__ == "__main__":
    main()