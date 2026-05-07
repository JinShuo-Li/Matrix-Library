from fastmatrix import Matrix, add, sub, transpose, matmul, hardware_threads, __version__


EPS = 1e-9


def assert_matrix_equal(actual: Matrix, expected, name: str) -> None:
    actual_list = actual.tolist()
    if len(actual_list) != len(expected):
        raise AssertionError(f"{name}: row count mismatch: {len(actual_list)} != {len(expected)}")
    for i, (row_a, row_e) in enumerate(zip(actual_list, expected)):
        if len(row_a) != len(row_e):
            raise AssertionError(f"{name}: col count mismatch on row {i}: {len(row_a)} != {len(row_e)}")
        for j, (va, ve) in enumerate(zip(row_a, row_e)):
            if abs(va - ve) > EPS:
                raise AssertionError(
                    f"{name}: value mismatch at ({i}, {j}): {va} != {ve}"
                )
    print(f"[PASS] {name}")



def assert_true(cond: bool, name: str) -> None:
    if not cond:
        raise AssertionError(f"{name}: condition is False")
    print(f"[PASS] {name}")



def assert_raises(func, exc_type, name: str) -> None:
    try:
        func()
    except exc_type:
        print(f"[PASS] {name}")
        return
    except Exception as e:
        raise AssertionError(
            f"{name}: expected {exc_type.__name__}, got {type(e).__name__}: {e}"
        ) from e
    raise AssertionError(f"{name}: expected {exc_type.__name__}, but no exception was raised")



def run_basic_info_tests() -> None:
    print("=== Basic package info ===")
    print("fastmatrix version:", __version__)
    threads = hardware_threads()
    print("hardware_threads():", threads)
    assert_true(isinstance(__version__, str) and len(__version__) > 0, "__version__ is a non-empty string")
    assert_true(isinstance(threads, int) and threads >= 1, "hardware_threads() returns int >= 1")



def run_constructor_and_property_tests() -> tuple[Matrix, Matrix, Matrix, Matrix]:
    print("\n=== Constructor / property tests ===")

    a = Matrix([[1, 2, 3], [4, 5, 6]])
    b = Matrix([[7, 8], [9, 10], [11, 12]])
    c = Matrix([[10, 20, 30], [40, 50, 60]])
    d = Matrix([[1, 2], [3, 4]])

    assert_true(a.shape == (2, 3), "Matrix.shape for A")
    assert_true(b.shape == (3, 2), "Matrix.shape for B")
    assert_matrix_equal(a, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "Matrix.tolist() for A")
    assert_matrix_equal(a.T, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], "Matrix.T")

    a_copy = a.copy()
    assert_matrix_equal(a_copy, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "Matrix.copy()")

    a_clone = Matrix(a)
    assert_matrix_equal(a_clone, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "Matrix(Matrix) clone constructor")

    rep = repr(a)
    print("repr(A):", rep)
    assert_true("Matrix(shape=(2, 3)" in rep, "Matrix.__repr__ contains shape")

    return a, b, c, d



def run_matrix_method_tests(a: Matrix, b: Matrix, c: Matrix, d: Matrix) -> None:
    print("\n=== Matrix method tests ===")

    assert_matrix_equal(a.add(c), [[11.0, 22.0, 33.0], [44.0, 55.0, 66.0]], "Matrix.add()")
    assert_matrix_equal(c.sub(a), [[9.0, 18.0, 27.0], [36.0, 45.0, 54.0]], "Matrix.sub()")
    assert_matrix_equal(a.matmul(b), [[58.0, 64.0], [139.0, 154.0]], "Matrix.matmul() default")
    assert_matrix_equal(
        a.matmul(b, threads=2, block_size=16),
        [[58.0, 64.0], [139.0, 154.0]],
        "Matrix.matmul(threads=2, block_size=16)",
    )
    assert_matrix_equal(d.matmul(d, threads=1, block_size=8), [[7.0, 10.0], [15.0, 22.0]], "Matrix.matmul() small square")



def run_operator_tests(a: Matrix, b: Matrix, c: Matrix, d: Matrix) -> None:
    print("\n=== Operator overload tests ===")

    assert_matrix_equal(a + c, [[11.0, 22.0, 33.0], [44.0, 55.0, 66.0]], "operator +")
    assert_matrix_equal(c - a, [[9.0, 18.0, 27.0], [36.0, 45.0, 54.0]], "operator -")
    assert_matrix_equal(a @ b, [[58.0, 64.0], [139.0, 154.0]], "operator @")
    assert_matrix_equal(2 * d, [[2.0, 4.0], [6.0, 8.0]], "operator __rmul__")
    assert_matrix_equal(d * 3, [[3.0, 6.0], [9.0, 12.0]], "operator __mul__")



def run_top_level_function_tests(a: Matrix, b: Matrix, c: Matrix) -> None:
    print("\n=== Top-level function tests ===")

    assert_matrix_equal(add(a, c), [[11.0, 22.0, 33.0], [44.0, 55.0, 66.0]], "fastmatrix.add()")
    assert_matrix_equal(sub(c, a), [[9.0, 18.0, 27.0], [36.0, 45.0, 54.0]], "fastmatrix.sub()")
    assert_matrix_equal(transpose(a), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], "fastmatrix.transpose() default")
    assert_matrix_equal(
        transpose(a, threads=2),
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
        "fastmatrix.transpose(threads=2)",
    )
    assert_matrix_equal(matmul(a, b), [[58.0, 64.0], [139.0, 154.0]], "fastmatrix.matmul() default")
    assert_matrix_equal(
        matmul(a, b, threads=2, block_size=16),
        [[58.0, 64.0], [139.0, 154.0]],
        "fastmatrix.matmul(threads=2, block_size=16)",
    )



def run_error_tests(a: Matrix, b: Matrix, d: Matrix) -> None:
    print("\n=== Error handling tests ===")

    assert_raises(lambda: Matrix([[1, 2], [3]]), ValueError, "ragged rows raise ValueError")
    assert_raises(lambda: a.add([[1, 2, 3]]), TypeError, "Matrix.add(non-Matrix) raises TypeError")
    assert_raises(lambda: a.sub([[1, 2, 3]]), TypeError, "Matrix.sub(non-Matrix) raises TypeError")
    assert_raises(lambda: a.matmul([[1], [2], [3]]), TypeError, "Matrix.matmul(non-Matrix) raises TypeError")
    assert_raises(lambda: a + [[1, 2, 3]], TypeError, "operator +(non-Matrix) raises TypeError")
    assert_raises(lambda: a - [[1, 2, 3]], TypeError, "operator -(non-Matrix) raises TypeError")
    assert_raises(lambda: a @ [[1], [2], [3]], TypeError, "operator @(non-Matrix) raises TypeError")
    assert_raises(lambda: add(a, d), RuntimeError, "shape mismatch in add() raises backend error")
    assert_raises(lambda: matmul(a, a), RuntimeError, "shape mismatch in matmul() raises backend error")



def main() -> None:
    run_basic_info_tests()
    a, b, c, d = run_constructor_and_property_tests()
    run_matrix_method_tests(a, b, c, d)
    run_operator_tests(a, b, c, d)
    run_top_level_function_tests(a, b, c)
    run_error_tests(a, b, d)
    print("\nAll interface tests passed.")


if __name__ == "__main__":
    main()
