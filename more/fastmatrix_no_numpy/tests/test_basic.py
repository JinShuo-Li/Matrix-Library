from fastmatrix import Matrix, hardware_threads


def test_basic_ops():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])

    assert (a + b).tolist() == [[6.0, 8.0], [10.0, 12.0]]
    assert (b - a).tolist() == [[4.0, 4.0], [4.0, 4.0]]
    assert a.T.tolist() == [[1.0, 3.0], [2.0, 4.0]]
    assert (2 * a).tolist() == [[2.0, 4.0], [6.0, 8.0]]


def test_matmul_and_threads():
    a = Matrix([[1, 2, 3], [4, 5, 6]])
    b = Matrix([[7, 8], [9, 10], [11, 12]])

    assert (a @ b).tolist() == [[58.0, 64.0], [139.0, 154.0]]
    assert a.matmul(b, threads=2, block_size=16).tolist() == [[58.0, 64.0], [139.0, 154.0]]
    assert hardware_threads() >= 1
