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

    This is a convenience wrapper around the ``Array.__matmul__()`` operator, similar to
    NumPy's ``np.matmul``. It multiplies two ``Array`` instances if both inputs
    are ``Array`` with compatible dimensions.

    Args:
        a: Left-hand side operand. Must be an ``Array``.
        b: Right-hand side operand. Must be an ``Array`` with dimensions compatible with ``a`` (i.e., ``a.shape[-1] == b.shape[-2]``).

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


def det(a: Array, algo: str = "gaussian") -> float:
    """
    Compute the determinant of a square 2D ``Array``.

    Args:
        a: Input array. Must be a 2D square matrix.

    Returns: Determinant of the matrix. Returns None if the input is not a square 2D Array.
    """
    if not isinstance(a, Array):
        return
    if a.ndim != 2 or not a.is_square():
        return

    match algo:
        case "laplace":
            return det_laplace(a)
        case "LU":
            return det_LU(a)
        case "gaussian":
            return det_gaussian(a)
        case _:
            return det_gaussian(a)


# O(n!)
def det_laplace(a: Array) -> float:
    """
    Compute the determinant of a square 2D ``Array`` using recursive expansion (Laplace expansion along the first column).

    Args:
        a: Input array. Must be a 2D square matrix.

    Time complexity: O(n!) where n is the dimension of the square matrix ``a``

    Space complexity: O(n^2)

    Returns: Determinant of the matrix. Returns None if the input is not a square 2D Array.
    """
    if not isinstance(a, Array):
        return
    if a.ndim != 2 or not a.is_square():
        return
    n = a.shape[0]
    if n == 1:
        return a._data[0]

    data = a.data

    ans = 0
    for i in range(n):
        cooef = Array([
            data[x][y]
            for x in range(n)
            for y in range(1, n)
            if x != i
        ], shape=(n - 1, n - 1))
        ans += (-1)**i * data[i][0] * det_laplace(cooef)
    return ans

# O(n^3)


def det_gaussian(a: Array) -> float:
    """
    Compute the determinant of a square 2D ``Array`` using Gaussian elimination.
    This algorithm performs row-reduction to upper triangular form, tracking
    row swaps and multiplying the diagonal entries to obtain the determinant.

    Args:
        a: Input array. Must be a 2D square matrix.

    Time complexity: O(n^3) where n is the dimension of the square matrix ``a``

    Space complexity: O(n^2)

    Returns: Determinant of the matrix. Returns None if the input is not a square 2D Array.
    """
    if not isinstance(a, Array):
        return
    if not a.is_square():
        return

    data = a.data

    n = a.shape[0]
    row_indices = list(range(n))

    prod = 1
    for k in range(n):

        if data[row_indices[k]][k] == 0:
            for j in range(k + 1, n):
                if data[row_indices[j]][k] != 0:
                    # swap here, swap indice is better that swapping the whole list
                    tmp = row_indices[k]
                    row_indices[k] = row_indices[j]
                    row_indices[j] = tmp
                    prod *= -1
                    break

        current_row = data[row_indices[k]]
        akk = current_row[k]

        for i in range(k + 1, n):
            aik = data[row_indices[i]][k]
            for l in range(n):
                data[row_indices[i]][l] -= aik / akk * current_row[l]

    for k in range(n):
        prod *= data[row_indices[k]][k]
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


def det_LU(A):
    """
    Compute the determinant of a square 2D ``Array`` using LU decomposition.

    This function factorizes the matrix A into L and U (via ``LU_decomposition``),
    then computes the determinant as the product of the diagonal entries of U.
    Since det(A) = det(L) * det(U) and det(L) = 1 for a unit lower triangular L,
    the determinant is simply the product of U's diagonal.

    Args:
        A: Input 2D ``Array`` (matrix). Must be square.

    Time complexity: O(n^3)

    Space complexity: O(n^2)

    Returns: Determinant of the matrix. Returns None if the input is not a square 2D Array.
    """
    _, U = LU_decomposition(A)
    det = 1
    for i in range(len(U)):
        det *= U[i][i]
    return det


def forward_sub_inverse_LU(L, b):
    """
    TODO: Add description.

    Args:
        A:
        L:
        b:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    n = len(L)
    y = [0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    return y


def backward_sub_inverse_LU(U, y):
    """
    TODO: Add description.

    Args:
        A:
        U:
        y:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    n = len(U)
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]
    return x

# TODO: Optimize PLS


def inv(a: Array, algo: str = "LU") -> Array:
    """
    Compute the inverse of a square 2D ``Array``.

    Args:
        a: Input array. Must be a 2D square matrix.

    Returns: Inverse of the matrix. Returns None if the input is a non-invertible matrix.
    """
    if not isinstance(a, Array):
        return
    if a.ndim != 2 or not a.is_square():
        return

    match algo:
        case "recursion":
            return inverse_recursion(a)
        case "LU":
            return inverse_LU(a)
        case _:
            return inverse_LU(a)


def inverse_LU(a):
    """
    TODO: Add description.

    Args:
        a:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    if not isinstance(a, Array):
        return
    if not a.is_square():
        return

    identity = [[1 if i == j else 0 for j in range(
        a.shape[0])] for i in range(a.shape[0])]
    ans = []
    L, U = LU_decomposition(a)
    for i in range(a.shape[0]):
        ans.append(backward_sub_inverse_LU(
            U, forward_sub_inverse_LU(L, identity[i])))
    return Array(ans).transpose()


# O(n!)
def inverse_recursion(a: Array) -> Array:
    """
    TODO: Add description.

    Args:
        a:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    if not isinstance(a, Array):
        return
    if not a.is_square():
        return
    n = a.shape[0]

    cofactor = Array([
        (-1) ** (i + j) * det(Array([
            a.data[x][y]
            for x in range(n)
            for y in range(n)
            if x != i and y != j
        ], shape=(n - 1, n - 1)))
        for i in range(n)
        for j in range(n)], shape=a.shape)

    return cofactor.transpose() / det(a)

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
