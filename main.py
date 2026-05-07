import math
import random

try:
    import sympy as sp
except ImportError:
    sp = None

from matlibrary import Matrix


def _frob_norm(data):
    total = 0.0
    for row in data:
        for val in row:
            total += val * val
    return math.sqrt(total)


def _diff_frob(a, b):
    if not a:
        return 0.0
    total = 0.0
    rows = len(a)
    cols = len(a[0])
    for i in range(rows):
        for j in range(cols):
            diff = a[i][j] - b[i][j]
            total += diff * diff
    return math.sqrt(total)


def _residual_error(A, Q, R):
    A_data = A.to_float_data()
    QR_data = (Q * R).to_float_data()
    return _diff_frob(A_data, QR_data)


def _orthogonality_error(Q):
    m, _ = Q.dim
    QtQ = Q.transpose() * Q
    I = Matrix.identity(m, tolerance=Q.tolerance, dtype="float")
    return _diff_frob(QtQ.to_float_data(), I.to_float_data())


def _sympy_qr(data):
    if sp is None:
        print("sympy is not installed. Install with: pip install sympy")
        return None
    A_sym = sp.Matrix(data)
    Q_sym, R_sym = A_sym.QRdecomposition()
    return A_sym, Q_sym, R_sym


def _sympy_echelon(data):
    """Return sympy row echelon form (not reduced) of a matrix."""
    if sp is None:
        return None
    A_sym = sp.Matrix(data)
    return A_sym.echelon_form()


def _format_matrix(data, precision=6, width=12):
    if not data:
        return "[]"
    lines = []
    for row in data:
        parts = [f"{val:>{width}.{precision}f}" for val in row]
        lines.append("[" + " ".join(parts) + "]")
    return "\n".join(lines)


def _sympy_to_float_list(sympy_mat):
    rows, cols = sympy_mat.shape
    return [[float(sympy_mat[i, j]) for j in range(cols)] for i in range(rows)]


def main():
    print("=" * 60)
    print("Task 1")

    rng = random.Random(0)
    rows, cols = 10, 8
    data = [[rng.randint(-9, 9) for _ in range(cols)] for _ in range(rows)]
    A = Matrix(data, tolerance=1e-10, dtype="float")

    print("Original matrix A (10x8):")
    print(A)
    print()

    ref, swaps = A.gauss_elimination(pivoting=True)
    print("Row echelon form (pivoting=True):")
    print(ref)
    print(f"Number of row swaps: {swaps}")
    print()

    if sp is not None:
        sympy_ech = _sympy_echelon(data)
        if sympy_ech is not None:
            print("Sympy row echelon form (for comparison):")
            print(_format_matrix(_sympy_to_float_list(sympy_ech), precision=4, width=10))
            our_rank = ref.rank()
            sympy_rank = sympy_ech.rank()
            print(f"Our rank: {our_rank}, Sympy rank: {sympy_rank}")
    else:
        print("sympy not available for verification.")
    print()

    print("=" * 60)
    print("Task 2")

    A2_data = [[3, 14, 9],
               [6, 43, 3],
               [6, 22, 15]]
    A2 = Matrix(A2_data, tolerance=1e-10, dtype="float")
    Q2_h, R2_h = A2.qr_decomposition(method="householder")

    print("Matrix A:")
    print(A2)
    print()
    print("Householder Q:")
    print(Q2_h)
    print()
    print("Householder R:")
    print(R2_h)
    print()

    err_h = _residual_error(A2, Q2_h, R2_h)
    orth_h = _orthogonality_error(Q2_h)
    print(f"Householder: residual={err_h:.6e}, orthogonality={orth_h:.6e}")
    print()

    sympy_result2 = _sympy_qr(A2_data)
    if sympy_result2 is not None:
        A2_sym, Q2_sym, R2_sym = sympy_result2
        diff = A2_sym - Q2_sym * R2_sym
        err_sym = math.sqrt(sum(float(x) * float(x) for x in diff))
        print("Sympy Q:")
        print(_format_matrix(_sympy_to_float_list(Q2_sym), precision=4, width=12))
        print("Sympy R:")
        print(_format_matrix(_sympy_to_float_list(R2_sym), precision=4, width=12))
        print(f"Sympy QR:    residual={err_sym:.6e}")
    print()

    print("=" * 60)
    print("Task 3")

    A3_data = [[2, 2, 1],
               [0, 2, 2],
               [2, 1, 2]]
    A3 = Matrix(A3_data, tolerance=1e-10, dtype="float")
    Q3_g, R3_g = A3.qr_decomposition(method="givens")

    print("Matrix A:")
    print(A3)
    print()
    print("Givens Q:")
    print(Q3_g)
    print()
    print("Givens R:")
    print(R3_g)
    print()

    err_g = _residual_error(A3, Q3_g, R3_g)
    orth_g = _orthogonality_error(Q3_g)
    print(f"Givens:      residual={err_g:.6e}, orthogonality={orth_g:.6e}")
    print()

    # sympy 验证
    sympy_result3 = _sympy_qr(A3_data)
    if sympy_result3 is not None:
        A3_sym, Q3_sym, R3_sym = sympy_result3
        diff = A3_sym - Q3_sym * R3_sym
        err_sym = math.sqrt(sum(float(x) * float(x) for x in diff))
        print("Sympy Q:")
        print(_format_matrix(_sympy_to_float_list(Q3_sym), precision=4, width=12))
        print("Sympy R:")
        print(_format_matrix(_sympy_to_float_list(R3_sym), precision=4, width=12))
        print(f"Sympy QR:    residual={err_sym:.6e}")
    print()


if __name__ == "__main__":
    main()