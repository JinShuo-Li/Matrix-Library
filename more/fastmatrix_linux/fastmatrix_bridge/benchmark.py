import time
import numpy as np
from fastmatrix import matmul

m = n = k = 512
A = np.random.rand(m, n)
B = np.random.rand(n, k)

for threads in [1, 2, 4, 8]:
    t0 = time.perf_counter()
    C = matmul(A, B, threads=threads, block_size=64)
    dt = time.perf_counter() - t0
    print(f"threads={threads:<2d} time={dt:.4f}s checksum={C.sum():.6f}")
