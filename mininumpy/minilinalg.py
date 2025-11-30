from mininumpy.array import *
import random


def dot(a: Array, b: Array) -> Array:
    """
    Perform matrix multiplication between two ``Array`` objects.

    This is a convenience wrapper around the ``Array.__matmul__()`` operator, similar to
    NumPy's ``np.matmul``. It multiplies two ``Array`` instances if both inputs
    are ``Array`` with compatible dimensions.

    Args:
        a: Left-hand side operand. Must be an ``Array``.
        b: Right-hand side operand. Must be an ``Array`` with dimensions compatible with ``a`` (i.e., ``a.shape[-1] == b.shape[-2]``).

    Returns: New Array containing the matrix product. Returns None if the operation is invalid.
    """
    if isinstance(a, Array) and isinstance(b, Array):
        return a @ b
    return


def matmul(a: Array, b: Array, algo="naive") -> Array:
    """
    Perform matrix multiplication between two ``Array`` objects.

    Args:
        a: Left-hand side operand. Must be an ``Array``.
        b: Right-hand side operand. Must be an ``Array`` with dimensions compatible with ``a`` (i.e., ``a.shape[-1] == b.shape[-2]``).
        algo: String identifier for the algorithm used. Could be: ``naive`` (default) | ``strassen``.

    Returns: New Array containing the matrix product. Returns None if the operation is invalid.
    """
    if not isinstance(b, Array):
        return
    if a.shape[-1] != b.shape[-2]:
        return

    match algo:
        case "naive":
            return a @ b
        case "strassen":
            mul = matmul_flat_2D_strassen
        case _:
            return a @ b

    count = math.prod(a.shape[:-2])

    _data_ans = []

    n = a.shape[-2]
    p = a.shape[-1]
    m = b.shape[-1]

    # O(count * (ndim + n * m * p)) where count  = size / (n * p)
    for c in range(count):
        # O(ndim)
        start_a = c * n * p
        stop_a = start_a + n * p

        start_b = c * m * p
        stop_b = start_b + m * p

        # O(n * m * p)
        _data_ans += mul(a._data[start_a:stop_a],
                         b._data[start_b:stop_b],
                         n, p, m)

    new_shape = list(a.shape)
    new_shape[-1] = m
    return Array(_data_ans, shape=tuple(new_shape))


def norm(a: Array) -> float:
    """
    Compute the Euclidean (L2) norm of an ``Array``.

    This calculates the square root of the sum of squares of all elements, i.e. ||a|| = sqrt(sum(e²)).

    Args:
        a: Input ``Array``.

    Time complexity: O(size)

    Space complexity: O(1)

    Returns: The Euclidean norm of the array. Returns None if the input is not an ``Array``.
    """
    if not isinstance(a, Array):
        return
    ans = 0
    for e in a._data:
        ans += abs(e) ** 2
    return math.sqrt(ans)


def det(a: Array, algo: str = "gaussian") -> float | list[float]:
    """
    Compute the determinant of a square 2D ``Array``.

    Args:
        a: Input array. Must be a 2D square matrix.
        algo: String identifier for the algorithm used. Could be: ``laplace`` | ``LU`` |  ``gaussian`` (default).

    Returns: Determinant of the matrix. Returns None if the input is not a square 2D Array.
    """
    if not isinstance(a, Array):
        return
    if a.shape[-1] != a.shape[-2]:
        return

    match algo:
        case "laplace":
            run = det_laplace
        case "LU":
            run = det_LU
        case "gaussian":
            run = det_gaussian
        case _:
            run = det_gaussian

    count = math.prod(a.shape[:-2])

    if count == 1:
        return run(a._data, a.shape[-1])

    det_list = []

    n = a.shape[-2]

    for c in range(count):
        start_a = c * n * n
        stop_a = start_a + n * n
        det_list.append(run(a._data[start_a: stop_a], n))

    return Array(det_list, shape=a.shape[:-2])


# O(n!)
def det_laplace(a: list[float], n: int) -> float:
    """
    Compute the determinant of a flatten square 2D matrix using recursive expansion (Laplace expansion along the first column).

    Args:
        a: Input flat array. Must be a 2D square matrix.
        n: The matrix dimension.

    Time complexity: O(n!)

    Space complexity: O(n^2)

    Returns: Determinant of the matrix. Returns None if the input is not square.
    """
    if len(a) != n * n:
        return
    A = Array(a, shape=(n, n))

    if n == 1:
        return A._data[0]

    ans = 0
    for i in range(n):
        cooef = [
            A[x, y]
            for x in range(n)
            for y in range(1, n)
            if x != i
        ]
        ans += (-1)**i * A[i, 0] * det_laplace(cooef, n - 1)
    return ans

# O(n^3)


def det_gaussian(a: list[float], n: int) -> float:
    """
    Compute the determinant of a flat square 2D matrix using Gaussian elimination.
    This algorithm performs row-reduction to upper triangular form, tracking
    row swaps and multiplying the diagonal entries to obtain the determinant.

    Args:
        a: Input flat array. Must be a 2D square matrix.
        n: The matrix dimension.

    Time complexity: O(n^3)

    Space complexity: O(n^2)

    Returns: Determinant of the matrix. Returns None if the input is not square.
    """
    A = Array(a, shape=(n, n))

    row_indices = list(range(n))

    prod = 1
    for k in range(n):

        if A[row_indices[k], k] == 0:
            for j in range(k + 1, n):
                if A[row_indices[j], k] != 0:
                    # swap here, swap indice is better that swapping the whole list
                    tmp = row_indices[k]
                    row_indices[k] = row_indices[j]
                    row_indices[j] = tmp
                    prod *= -1
                    break

        akk = A[row_indices[k], k]

        for i in range(k + 1, n):
            aik = A[row_indices[i], k]
            for l in range(n):
                A[row_indices[i], l] -= aik / akk * A[row_indices[k], l]

    for k in range(n):
        prod *= A[row_indices[k], k]
    return prod


def LU_decomposition(A: Array) -> tuple[list[list[float]], list[list[float]]]:
    """
    Perform LU decomposition of a 2D ``Array`` (matrix).

    This factorizes a square matrix A into the product of a lower triangular
    matrix L and an upper triangular matrix U, such that A = L * U.

    - L is a unit lower triangular matrix (diagonal entries are 1).

    - U is an upper triangular matrix.

    Args:
        A: Input 2D ``Array`` (matrix). Must be two-dimensional.

    Time complexity: O(n^3)

    Space complexity: O(n^2)

    Returns: A tuple (L, U) where:

            - L is the unit lower triangular matrix.

            - U is the upper triangular matrix.

        Returns None if the input is not a 2D ``Array``.
    """
    if not isinstance(A, Array):
        return

    if A.ndim != 2:
        return

    L = [[1 if i == j else 0 for j in range(
        A.shape[0])] for i in range(A.shape[0])]

    U = []
    for i in range(A.shape[0]):
        row = []
        for j in range(A.shape[1]):
            row.append(A.element_type(A.data[i][j]))
        U.append(row)

    for i in range(A.shape[0] - 1):
        for j in range(i + 1, A.shape[0]):
            L[j][i] = U[j][i] / U[i][i]
            for k in range(A.shape[1]):
                U[j][k] -= U[i][k] * L[j][i]

    return (L, U)


def det_LU(a: list[float], n: int) -> float:
    """
    Compute the determinant of a flat square 2D matrix using LU decomposition.

    Args:
        a: Input flat 2D matrix. Must be square.
        n: The matrix dimension.

    Time complexity: O(n^3)

    Space complexity: O(n^2)

    Returns: Determinant of the matrix. Returns None if the input is not square.
    """
    _, U = LU_decomposition(Array(a, shape=(n, n)))
    det = 1
    for i in range(len(U)):
        det *= U[i][i]
    return det


def _forward_sub_inverse_LU(L, b):
    n = len(L)
    y = [0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    return y


def _backward_sub_inverse_LU(U, y):
    n = len(U)
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]
    return x


def inv(a: Array, algo: str = "LU") -> Array:
    """
    Compute the inverse of a square 2D ``Array``.

    Args:
        a: Input array. Must be a 2D square matrix.
        algo: String identifier for the algorithm used. Could be: ``recursion`` | ``LU`` (default) |  ``gaussian``.

    Returns: Inverse of the matrix. Returns None if the input is a non-invertible matrix.
    """
    if not isinstance(a, Array):
        return
    if a.shape[-1] != a.shape[-2]:
        return

    match algo:
        case "recursion":
            inv = inverse_recursion
        case "LU":
            inv = inverse_LU
        case "gaussian":
            inv = inverse_gaussian
        case _:
            inv = inverse_LU

    count = math.prod(a.shape[:-2])

    inv_list = []

    n = a.shape[-2]

    for c in range(count):
        start_a = c * n * n
        stop_a = start_a + n * n
        inv_list += inv(a._data[start_a: stop_a], n)

    return Array(inv_list, shape=a.shape)


def inverse_LU(a: list[float], n: int) -> list[float]:
    """
    Compute the inverse of a flat square 2D matrix using LU decomposition.

    Args:
        a: Flat list representing the matrix entries row-major.
        n: Dimension of the square matrix ``a``.

    Time complexity: O(n^3)

    Space complexity: O(n^2)

    Returns: Flattened list representing the inverse matrix in row-major order.
              Returns None if the input is not a square 2D Array.
    """
    if len(a) != n * n:
        return
    A = Array(a, shape=(n, n))

    identity = [[1 if i == j else 0 for j in range(
        A.shape[0])] for i in range(A.shape[0])]
    ans = []
    L, U = LU_decomposition(A)
    for i in range(A.shape[0]):
        ans.append(_backward_sub_inverse_LU(
            U, _forward_sub_inverse_LU(L, identity[i])))
    return Array(ans).transpose()._data


# O(n!)
def inverse_recursion(a, n):
    """
    Compute the inverse of a square 2D ``Array`` using recursive cofactor expansion.

    Args:
        a: Flat list representing the matrix entries in row-major order.
        n: Dimension of the square matrix ``a``.

    Time complexity: O(n!) 

    Space complexity: O(n^2)

    Returns: Flattened list representing the inverse matrix in row-major order.
    """
    A = Array(a, shape=(n, n))

    cofactor = Array([
        (-1) ** (i + j) * det(Array([
            A[x, y]
            for x in range(n)
            for y in range(n)
            if x != i and y != j
        ], shape=(n - 1, n - 1)))
        for i in range(n)
        for j in range(n)], shape=A.shape)

    return (cofactor.transpose() / det(A))._data


def inverse_gaussian(a, n):
    """
    Compute the inverse of a square 2D ``Array`` using Gauss-Jordan elimination with partial pivoting.

    Args:
        a: Flat list representing the matrix entries in row-major order.
        n: Dimension of the square matrix ``a``.

    Time complexity: O(n^3)

    Space complexity: O(n^2)

    Returns:
        list: Flattened list representing the inverse matrix in row-major order.
    """
    A = Array(a, shape=(n, n))
    I = eye(n)

    row_indices = list(range(n))

    for k in range(n):

        if A[row_indices[k], k] == 0:
            for j in range(k + 1, n):
                if A[row_indices[j], k] != 0:
                    tmp = row_indices[k]
                    row_indices[k] = row_indices[j]
                    row_indices[j] = tmp
                    break

        pivot = A[row_indices[k], k]

        for l in range(n):
            A[row_indices[k], l] /= pivot
            I[row_indices[k], l] /= pivot

        for i in range(k + 1, n):
            factor = A[row_indices[i], k]
            if factor != 0:
                for l in range(n):
                    A[row_indices[i], l] -= factor * A[row_indices[k], l]
                    I[row_indices[i], l] -= factor * I[row_indices[k], l]

    for k in range(n - 1, -1, -1):
        for i in range(k - 1, -1, -1):
            factor = A[row_indices[i], k]
            if factor != 0:
                for l in range(n):
                    A[row_indices[i], l] -= factor * A[row_indices[k], l]
                    I[row_indices[i], l] -= factor * I[row_indices[k], l]

    result = Array([0] * n * n, shape=(n, n))
    for i in range(n):
        for j in range(n):
            result[i, j] = I[row_indices[i], j]

    return result._data


# # O(n^2)
# def power_iteration(a, n, max_iter: int = 1000, eps: float = 1e-6):
#     A = Array(a, shape=(n, n))

#     v = Array([[random.random() for _ in range(n)]]).transpose()

#     for _ in range(max_iter):
#         w = A @ v

#         w_norm = norm(w)
#         if w_norm == 0:
#             break

#         w = w / w_norm


#         if (abs(v - w)).max() < eps:
#             v = w
#             break

#         v = w

#     lam = (v.conj().transpose() @ A @ v)[0, 0]

#     return lam, v


# def eig_flat_2D_power_iteration(a, n, max_iter=1000, eps=1e-6):

#     M = Array(a, shape=(n, n))  # make a working copy
#     lamda = [0] * n
#     VT = []

#     for i in range(n):

#         lam, v = power_iteration(M._data, n, max_iter=max_iter, eps=eps)

#         lamda[i] = lam

#         v_norm = v / norm(v)
#         VT.append(v_norm.transpose()._data)

#         M = M - lam * (v_norm @ v_norm.conj().transpose())

#     return lamda, Array(VT).transpose().data

# def eig(a, max_iter=1000, eps=1e-6):
#     if not isinstance(a, Array):
#         return
#     if a.shape[-1] != a.shape[-2]:
#         return
#     n = a.shape[-1]

#     count = math.prod(a.shape[:-2])

#     eig_values = Array([0] * (count * n), shape=a.shape[:-1])
#     eig_vectors = Array([0] * a.size, shape=a.shape)

#     n = a.shape[-2]

#     for c in range(count):
#         start_a = c * (n * n)
#         stop_a  = start_a + (n * n)

#         # O(n ^ 3)
#         l, v = eig_flat_2D_power_iteration(
#             a._data[start_a:stop_a], n, max_iter, eps)

#         start_val = c * n
#         stop_val = start_val + n
#         eig_values._data[start_val:stop_val] = l
#         eig_vectors._data[start_a:stop_a] = flatten(v)

#     return eig_values, eig_vectors
