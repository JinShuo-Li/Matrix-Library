import numpy as np

from fastmatrix import Matrix, add, sub, transpose, matmul


def test_basic_ops():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert np.allclose(add(a, b), a + b)
    assert np.allclose(sub(a, b), a - b)
    assert np.allclose(transpose(a), a.T)
    assert np.allclose(matmul(a, b), a @ b)


def test_matrix_wrapper():
    a = Matrix([[1, 2, 3], [4, 5, 6]])
    b = Matrix([[7, 8], [9, 10], [11, 12]])
    c = a @ b
    assert np.allclose(c.numpy(), np.array([[58.0, 64.0], [139.0, 154.0]]))

print("All tests passed!")