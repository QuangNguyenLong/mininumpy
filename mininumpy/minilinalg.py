from mininumpy.array import *


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


def matmul(a: Array, b: Array) -> Array:
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
        ans += e * e
    return math.sqrt(ans)

# TODO: Optimize PLS
# O(n!)


def det(a: Array) -> float:
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
        ans += (-1)**i * data[i][0] * det(cooef)
    return ans

# O(n^3)


def det_gaussian(a : Array) -> float:
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
    row_indices = range(n)

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


def LU_decomposition(A : Array) -> tuple[list[list[float]], list[list[float]]]:
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


def inverse_LU(A):
    """
    TODO: Add description.

    Args:
        A:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    if not isinstance(A, Array):
        return
    if not A.is_square():
        return

    identity = [[1 if i == j else 0 for j in range(
        A.shape[0])] for i in range(A.shape[0])]
    ans = []
    L, U = LU_decomposition(A)
    for i in range(A.shape[0]):
        ans.append(backward_sub_inverse_LU(
            U, forward_sub_inverse_LU(L, identity[i])))
    return Array(ans).transpose()

# TODO: Optimize PLS
# O(n!)


def inv(a):
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

# TODO: complete eig value and VECTOR PLS, change algo to O(n^3) using TP2


def compute_rotation_coeff(a, b, d):
    tau = (d - a) / (2 * b)
    if tau < 0:
        sign_tau = -1
    else:
        sign_tau = 1
    t = sign_tau / (math.fabs(tau) + math.sqrt(1 + tau * tau))
    c = 1 / (math.sqrt(1 + t * t))
    s = t * c
    return c, s


def jacobi_rotate(data, V, n):
    A = Array(data)
    B = Array(V)
    for i in range(n):
        for j in range(n):
            if i != j and A.data[i][j] != 0:
                # rotate only if the current element is not zero
                c, s = compute_rotation_coeff(
                    A.data[i][i], A.data[i][j], A.data[j][j])

                # do this instead of J^T @ mat @ T because this is O(n) instead of O(n^3)
                for k in range(n):
                    # mat = J^T @ mat @ T
                    a_ki = data[k][i]
                    a_kj = data[k][j]
                    data[k][i] = c * a_ki - s * a_kj
                    data[k][j] = s * a_ki + c * a_kj

                for k in range(n):
                    a_ik = data[i][k]
                    a_jk = data[j][k]
                    data[i][k] = c * a_ik - s * a_jk
                    data[j][k] = s * a_ik + c * a_jk

                # V @= J
                for k in range(n):
                    v_ki = V[k][i]
                    v_kj = V[k][j]
                    V[k][i] = c * v_ki - s * v_kj
                    V[k][j] = s * v_ki + c * v_kj
    return A.data, B.data


def eig(mat, max_iter=1000, eps=1e-6):
    """
    TODO: Add description.

    Args:
        a:
        max_iter:
        eps:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    if not isinstance(mat, Array):
        return
    if not mat.is_square():
        return

    n = mat.shape[0]
    data = mat.data.copy()
    V = eye(n).data.copy()

    # this is O(n^3)
    for _ in range(max_iter):

        old = Array(data)
        data, V = jacobi_rotate(data, V, n)
        # Check convergence
        new = Array(data)

        max_val = abs(old - new).max()
        # if the max changes is even smaller than eps, then why bother changes

        if max_val < eps:
            break

    return [data[i][i] for i in range(n)], V
