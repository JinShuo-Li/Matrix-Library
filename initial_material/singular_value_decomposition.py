from fractions import Fraction
import math

class Matrix:
    def __init__(self, matrix, tolerance=1e-10):
        self.matrix = [[Fraction(x) for x in row] for row in matrix]
        self.size = (len(matrix), len(matrix[0]) if matrix else 0)
        self.tolerance = tolerance

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
        # Format explicitly with a fixed width (e.g. 10 characters) and right-aligned (>), retaining 4 decimal places
        return '\n'.join(['[' + '\t'.join(f"{float(x):>10.4f}" for x in row) + ']' for row in self.matrix])
    
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
        
        r = 0
        for c in range(cols):
            if r >= rows:
                break

            pivot_row = r
            while pivot_row < rows and new_matrix[pivot_row][c] == 0:
                pivot_row += 1
            
            if pivot_row == rows:
                continue
            
            if r != pivot_row:
                new_matrix[r], new_matrix[pivot_row] = new_matrix[pivot_row], new_matrix[r]
                swap_count += 1
            
            for j in range(r + 1, rows):
                if new_matrix[j][c] != 0:
                    factor = new_matrix[j][c] / new_matrix[r][c]
                    for k in range(c, cols):
                        new_matrix[j][k] -= factor * new_matrix[r][k]
            r += 1
                        
        return Matrix(new_matrix), swap_count
    
    def eigenvalues(self):
        '''
        We will implement the Jacobi rotation method to find the eigenvalues and eigenvectors of a symmetric matrix.
        Returns a tuple: (list of eigenvalues, Matrix of eigenvectors as columns)
        '''
        tolerance = self.tolerance
        m, n = self.size
        if m != n:
            raise ValueError("Eigenvalues are only defined for square matrices.")

        A = [[float(self.matrix[i][j]) for j in range(n)] for i in range(m)]
        V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(m)]
        
        max_iterations = 1000
        for _ in range(max_iterations):
            max_val = 0.0
            p = -1
            q = -1
            for i in range(m):
                for j in range(i + 1, n):
                    if abs(A[i][j]) > max_val:
                        max_val = abs(A[i][j])
                        p = i
                        q = j

            if max_val < tolerance:
                break

            tau = (A[q][q] - A[p][p]) / (2.0 * A[p][q])

            sgn_tau = 1.0 if tau >= 0 else -1.0
            t = sgn_tau / (abs(tau) + math.sqrt(tau**2 + 1.0))
            
            c = 1.0 / math.sqrt(1.0 + t**2)
            s = t * c

            for i in range(m):
                if i != p and i != q:
                    A_ip = A[i][p]
                    A_iq = A[i][q]
                    A[i][p] = c * A_ip - s * A_iq
                    A[p][i] = A[i][p]
                    A[i][q] = c * A_iq + s * A_ip
                    A[q][i] = A[i][q]
                    
            A_pp = A[p][p]
            A_qq = A[q][q]
            A_pq = A[p][q]
            
            A[p][p] = c**2 * A_pp - 2.0 * s * c * A_pq + s**2 * A_qq
            A[q][q] = s**2 * A_pp + 2.0 * s * c * A_pq + c**2 * A_qq
            A[p][q] = 0.0
            A[q][p] = 0.0
            
            for i in range(m):
                V_ip = V[i][p]
                V_iq = V[i][q]
                V[i][p] = c * V_ip - s * V_iq
                V[i][q] = c * V_iq + s * V_ip

        eigenvals = [A[i][i] if abs(A[i][i]) > tolerance else 0.0 for i in range(m)]
        return eigenvals, Matrix(V)


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

    def rank(self):
        mat, _ = self.gauss_elimination()
        rank = 0
        for i in range(self.size[0]):
            if any(abs(mat.matrix[i][j]) > self.tolerance for j in range(self.size[1])):
                rank += 1
        return rank

    def singular_value_decomposition(self):
        '''
        We will implement the Singular Value Decomposition (SVD) using the eigenvalue decomposition of A^T A and A A^T.
        Sorts the singular values in descending order and calibrates signs to ensure A = U * Sigma * V^T.
        '''
        m, n = self.size
        A_T = self.transpose()
        A_T_A = A_T * self
        A_A_T = self * A_T
        
        eigenvalues_ATA, V = A_T_A.eigenvalues()
        eigenvalues_AAT, U = A_A_T.eigenvalues()

        sigmas_V = []
        for i in range(n):
            val = math.sqrt(max(0, eigenvalues_ATA[i]))
            vec = [V.matrix[j][i] for j in range(n)]
            sigmas_V.append((val, vec))

        sigmas_U = []
        for i in range(m):
            val = math.sqrt(max(0, eigenvalues_AAT[i]))
            vec = [U.matrix[j][i] for j in range(m)]
            sigmas_U.append((val, vec))

        sigmas_V.sort(key=lambda x: x[0], reverse=True)
        sigmas_U.sort(key=lambda x: x[0], reverse=True)

        V_sorted = [[sigmas_V[j][1][i] for j in range(n)] for i in range(n)]
        U_sorted = [[sigmas_U[j][1][i] for j in range(m)] for i in range(m)]
        U_matrix = Matrix(U_sorted)
        V_matrix = Matrix(V_sorted)

        k = min(m, n)
        singular_values = [sigmas_V[i][0] if i < len(sigmas_V) else 0.0 for i in range(k)]
        
        Sigma = [[0.0 for j in range(n)] for i in range(m)]
        for i in range(k):
            Sigma[i][i] = singular_values[i]
        Sigma_matrix = Matrix(Sigma)

        A_V = self * V_matrix
        for i in range(k):
            sigma = singular_values[i]
            if sigma > self.tolerance:
                idx_to_check = 0
                max_val_in_col = 0.0
                for j in range(m):
                    if abs(A_V.matrix[j][i]) > max_val_in_col:
                        max_val_in_col = abs(A_V.matrix[j][i])
                        idx_to_check = j
                
                expected_sign_val = A_V.matrix[idx_to_check][i]
                current_U_val = U_matrix.matrix[idx_to_check][i]
                if expected_sign_val * current_U_val < -self.tolerance:
                    for j in range(m):
                        U_matrix.matrix[j][i] *= -1.0
                        
        return U_matrix, Sigma_matrix, V_matrix.transpose()


def main():
    print("--- SVD Verification Test ---")
    A = Matrix([[3, 2, 2], [2, 3, -2]], tolerance=1e-10)
    print("Original Matrix A:")
    print(A)
    
    U, Sigma, V_T = A.singular_value_decomposition()
    
    print("\nLeft Singular Vectors (U):")
    print(U)
    
    print("\nSingular Value Matrix (Sigma):")
    print(Sigma)
    
    print("\nRight Singular Vectors Transposed (V^T):")
    print(V_T)
    
    print("\nReconstructed Matrix (U * Sigma * V^T) - Should equal A:")
    try:
        A_reconstructed = U * Sigma * V_T
        print(A_reconstructed)
    except Exception as e:
        print(f"Error during reconstruction: {e}")

if __name__ == "__main__":
    main()