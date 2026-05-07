from fastmatrix import Matrix, hardware_threads

A = Matrix([[1, 2, 3], [4, 5, 6]])
B = Matrix([[7, 8], [9, 10], [11, 12]])

print("hardware_threads =", hardware_threads())
print("A.shape =", A.shape)
print("B.shape =", B.shape)
print("A @ B =", (A @ B).tolist())
print("A.matmul(B, threads=4, block_size=32) =", A.matmul(B, threads=4, block_size=32).tolist())
print("A.T =", A.T.tolist())
print("2 * A =", (2 * A).tolist())
