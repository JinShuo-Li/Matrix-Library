import numpy as np
from fastmatrix import Matrix, hardware_threads

print("hardware_threads =", hardware_threads())

A = Matrix([[1, 2, 3], [4, 5, 6]])
B = Matrix([[7, 8], [9, 10], [11, 12]])
C = A.matmul(B, threads=4, block_size=32)

print("A @ B =")
print(C.numpy())
print("transpose =")
print(C.T.numpy())

# 与 NumPy 对齐验证
np_A = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
np_B = np.asarray([[7, 8], [9, 10], [11, 12]], dtype=np.float64)
print("equal to numpy:", np.allclose(C.numpy(), np_A @ np_B))
