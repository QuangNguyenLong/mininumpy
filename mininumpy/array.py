# ---------------------------PART 1----------------------------------#
# this class is not meant for logging and printing :v
import math

# ----------------------------my stuff---------------------------------#


def dot(v1: list[float], v2: list[float]) -> float:
    """
    Compute the dot product of two vectors.

    Args:
        v1: First input vector.
        v2: Second input vector. Must have the same length as ``v1``.

    Time complexity: O(n), where n is the length of the vectors.
    Space complexity: O(1)

    Returns: Dot product of the two vectors. If the vectors have different lengths, returns ``None``.
    """
    if len(v1) != len(v2):
        return
    n = len(v1)
    ans = 0
    for i in range(n):
        ans += v1[i] * v2[i]
    return ans


def _morton_index(r: int, c: int) -> int:
    z = 0
    shift = 0
    while r > 0 or c > 0:
        z |= (c & 1) << shift
        z |= (r & 1) << (shift + 1)
        r >>= 1
        c >>= 1
        shift += 2
    return z


def _morton_decode(z: int) -> tuple[int, int]:
    r = 0
    c = 0
    bit = 0
    while z > 0:
        c |= (z & 1) << bit
        r |= ((z >> 1) & 1) << bit
        z >>= 2
        bit += 1
    return r, c


def _to_morton_iterative(inp: list[float], n: int) -> list[float]:
    assert n & (n - 1) == 0, "n must be a power of two"
    out = [0.0] * (n * n)
    for r in range(n):
        for c in range(n):
            idx = _morton_index(r, c)
            out[idx] = inp[r * n + c]
    return out


def _from_morton_iterative(inp: list[float], n: int) -> list[float]:
    assert n & (n - 1) == 0, "n must be a power of two"
    out = [0.0] * (n * n)
    for z in range(n * n):
        r, c = _morton_decode(z)
        out[r * n + c] = inp[z]
    return out


def _strassen_pad_zeros(A: list[float], n: int, m: int, k: int) -> tuple[list[float], int]:
    ans = []
    for i in range(n):
        ans += A[i * m: (i + 1) * m] + [0] * (k - m)
    ans += [0] * k * (k - n)
    return ans


def _strassen_remove_zeros(A: list[float], n: int, m: int, k: int) -> list[float]:
    out = []
    for i in range(n):
        row_start = i * k
        out += A[row_start: row_start + m]
    return out


def _strassen_prepare(A: list[float], n: int) -> tuple[list[float]]:
    return _to_morton_iterative(A, n)


def _strassen_add(A, B):
    return [A[i] + B[i] for i in range(len(A))]


def _strassen_sub(A, B):
    return [A[i] - B[i] for i in range(len(A))]


def _strassen_mul(A, B, n):
    if n == 2:
        M1 = (A[0] + A[3]) * (B[0] + B[3])
        M2 = (A[2] + A[3]) * B[0]
        M3 = A[0] * (B[1] - B[3])
        M4 = A[3] * (B[2] - B[0])
        M5 = (A[0] + A[1]) * B[3]
        M6 = (A[2] - A[0]) * (B[0] + B[1])
        M7 = (A[1] - A[3]) * (B[2] + B[3])
        return [M1 + M4 - M5 + M7, M3 + M5, M2 + M4, M1 - M2 + M3 + M6]

    half = n // 2

    Q = (n*n) // 4

    A11 = A[0*Q: 1*Q]
    A12 = A[1*Q: 2*Q]
    A21 = A[2*Q: 3*Q]
    A22 = A[3*Q: 4*Q]

    B11 = B[0*Q: 1*Q]
    B12 = B[1*Q: 2*Q]
    B21 = B[2*Q: 3*Q]
    B22 = B[3*Q: 4*Q]

    M1 = _strassen_mul(_strassen_add(A11, A22), _strassen_add(B11, B22), half)
    M2 = _strassen_mul(_strassen_add(A21, A22), B11, half)
    M3 = _strassen_mul(A11, _strassen_sub(B12, B22), half)
    M4 = _strassen_mul(A22, _strassen_sub(B21, B11), half)
    M5 = _strassen_mul(_strassen_add(A11, A12), B22, half)
    M6 = _strassen_mul(_strassen_sub(A21, A11), _strassen_add(B11, B12), half)
    M7 = _strassen_mul(_strassen_sub(A12, A22), _strassen_add(B21, B22), half)

    C = _strassen_add(_strassen_sub(_strassen_add(M1, M4), M5), M7) + _strassen_add(M3, M5) + \
        _strassen_add(M2, M4) + \
        _strassen_add(_strassen_sub(_strassen_add(M1, M3), M2), M6)
    return C


def _next_pow2(x: int) -> int:
    k = 1
    while k < x:
        k <<= 1
    return k


def matmul_flat_2D_strassen(A: list[float], B: list[float], n: int, p: int, m: int) -> list[float]:
    """
    Perform 2D matrix Strassen multiplication in flattened form.

    This computes C = A @ B, where:
      - A is a (n * p) matrix stored row-major in a flat list.
      - B is a (m * p) matrix stored row-major in a flat list.
      - The result C is an (n * m) matrix stored row-major in a flat list.

    Args:
        A: Flattened list representing an (n * p) matrix.
        B: Flattened list representing a (p * m) matrix.
        n: Number of rows in A.
        p: Number of columns in A (and rows in B).
        m: Number of columns in B.

    Time complexity: O(k ^ log2(7)) where k = 2 ^ int(log2(max(n, p, m)))
    Space complexity: O(k ^ 2)

    Returns: Flattened list representing the (n * m) result matrix.
    """
    q = _next_pow2(max(n, p, m))
    A2 = _strassen_pad_zeros(A, n, p, q)
    B2 = _strassen_pad_zeros(B, p, m, q)

    A_mor = _strassen_prepare(A2, q)
    B_mor = _strassen_prepare(B2, q)

    C_mor = _strassen_mul(A_mor, B_mor, q)
    return _strassen_remove_zeros(_from_morton_iterative(C_mor, q), n, m, q)

# nxp x pxm


def matmul_flat_2D(A: list[float], BT: list[float], n: int, p: int, m: int) -> list[float]:
    """
    Perform 2D matrix naive multiplication in flattened form.

    This computes C = A @ B, where:
      - A is an (n * p) matrix stored row-major in a flat list.
      - BT is the transpose of B, i.e. a (m * p) matrix stored row-major in a flat list.
        (so each row of BT corresponds to a column of the original B).
      - The result C is an (n * m) matrix stored row-major in a flat list.

    Args:
        A: Flattened list representing an (n * p) matrix.
        BT: Flattened list representing a (m * p) matrix (transpose of B).
        n: Number of rows in A.
        p: Number of columns in A (and rows in B).
        m: Number of columns in B (rows in BT).

    Time complexity: O(n * m * p)
    Space complexity: O(n * m)

    Returns: Flattened list representing the (n * m) result matrix.
    """
    ans = [0] * (m * n)
    # O(n * m * p)
    for i in range(n):
        for j in range(m):
            # O(p) dot product
            ans[shaped_to_flat_index((i, j), (n, m))] = dot(
                A[i * p:(i+1) * p], BT[j * p:(j+1) * p])
    return ans


def transpose_new_index(index: int, shape: tuple[int], axis: list[int]) -> int:
    """
    TODO: Add description.

    Args:
        index:
        shape:
        axis:

    Time complexity: O(ndim)

    Space complexity: O(ndim)

    Returns:
    """
    # O(ndim)
    vec = flat_index_to_shaped(index, shape)

    new_vec = list(vec)
    new_shape = list(shape)
    # O(ndim)
    for i in range(len(axis)):
        new_vec[i] = vec[axis[i]]
        new_shape[i] = shape[axis[i]]
    # O(ndim)
    return shaped_to_flat_index(new_vec, new_shape)


def shaped_to_flat_index(vec: tuple[int], shape: tuple[int]) -> int:
    """
    Convert a multi-dimensional index into a flat index for the given shape.

    Args:
        vec: Multi-dimensional index.
        shape: Target shape.

    Time complexity: O(ndim)

    Space complexity: O(ndim)

    Returns: Flat index corresponding to the multi-dimensional index.
    """
    ndim = len(shape)

    # Precompute product of later dimensions
    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    ans = 0
    for i in range(ndim):
        ans += vec[i] * strides[i]

    return ans


def flat_index_to_shaped(index: int, shape: tuple[int]) -> tuple[int]:
    """
    Convert a flat index into a multi-dimensional index for the given shape.

    Args:
        index: Flat index.
        shape: Target shape.

    Time complexity: O(ndim)

    Space complexity: O(ndim)

    Returns: Multi-dimensional index corresponding to the flat index.
    """
    ndim = len(shape)

    # Precompute product of later dimensions
    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    ans = [0] * ndim
    for i in range(ndim):
        ans[i] = index // strides[i]
        index %= strides[i]

    return tuple(ans)

# TODO: add unit test


def flatten(lst: list) -> list:
    """
    Get the flatten version of a nested list.

    Args:
        lst: nested list to be flatten.

    Time complexity: O(size)

    Space complexity: O(size)

    Returns: A new flatten list.
    """
    if not isinstance(lst, list):
        return lst

    flat = []

    def dfs(sub):
        if isinstance(sub, list):
            for e in sub:
                dfs(e)
        else:
            flat.append(sub)

    dfs(lst)
    return flat


class Array:
    """
    Numpy-like array. Store the data in a flatten array.
    """

    _data = []
    """
    The raw flatten data of the :class:`~mininumpy.array.Array`.
    """

    element_type = None
    """
    The type of elements stored in the :class:`~mininumpy.array.Array`.
    """

    shape = ()
    """
    The dimensions of the :class:`~mininumpy.array.Array`.
    """

    ndim = 0
    """
    The number of dimensions of the :class:`~mininumpy.array.Array`.
    """

    size = 0
    """
    The total number of elements in the :class:`~mininumpy.array.Array`.
    """

    @property
    def data(self) -> list:
        """
        Method to get the original data.

        Time complexity: O(size * ndim)

        Space complexity: O(size)

        Returns: The original data in a (nested) list.
        """
        curr = self._data
        for c in self.shape[:0:-1]:  # ndim
            next = []
            for i in range(int(len(curr) / c)):  # size / shape[i] = size
                next.append(curr[i * c: (i + 1) * c])
            curr = next
        return curr

    def _size(self) -> int:
        """
        Compute the number of elements in the array.

        Time complexity: O(1)

        Space complexity: O(1)

        Returns: The number of elements in integer.
        """
        return len(self._data)

    def _shape(self, array: list) -> tuple[int]:
        """
        Compute the shape of ``array`` by following the first element at each level of nesting.

        Args:
            array: the input (nested) list needed to compute shape.

        Time complexity: O(ndim)

        Space complexity: O(ndim)

        Returns: The dimensions of ``array``.
        """
        shape = (len(array), )
        e = array[0]
        while isinstance(e, list):
            shape += (len(e), )
            e = e[0]
        return shape

    def _ndim(self) -> int:
        """
        TODO: Add description.

        Time complexity: O(1)

        Space complexity: O(1)

        Returns:
        """
        return len(self.shape)

    def __init__(self, data: list, shape: tuple[int] = None, element_type=None):
        """
        Initialize an :class:`~mininumpy.array.Array` object from nested list ``data``.

        Args:
            data: Input nested list.
            shape: Desired shape. If ``None``, inferred automatically.
            element_type: Type to cast elements. If ``None``, inferred from data.

        Time complexity: O(size + ndim)

        Space complexity: O(size)

        Returns: None
        """
        if not isinstance(data, list):
            raise ValueError("wtf bro")

        if shape:
            # O(size)
            self._data = flatten(data)
            # O(ndim)
            size = math.prod(shape)
            if len(self._data) != size:
                raise ValueError(
                    f"Error: Can not change size. From {len(self._data)} to {size}.")
            self.shape = tuple(shape)
        else:
            # O(ndim)
            self.shape = self._shape(data)
            # O(size)
            self._data = flatten(data)

        self.ndim = self._ndim()
        self.size = self._size()

        if element_type:
            self.element_type = element_type
            # O(size)
            for i in range(self.size):
                self._data[i] = element_type(self._data[i])
        else:
            self.element_type = type(self._data[0])
        return

    def _format_recursive(self, flat, shape, idx=0, indent=0):
        if len(shape) == 1:
            # Base case: 1D slice -> print elements
            start = idx
            end = idx + shape[0]
            items = flat[start:end]
            return "[" + " ".join(str(x) for x in items) + "]"

        # Recursive case
        dim = shape[0]
        subshape = shape[1:]
        jump = 1
        for s in subshape:
            jump *= s

        lines = []
        for i in range(dim):
            sub_idx = idx + i * jump
            sub = self._format_recursive(flat, subshape, sub_idx, indent + 2)
            if i == 0:
                lines.append("[" + sub)
            else:
                lines.append(" " * (indent + 1) + sub)
        lines[-1] += "]"
        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Recursive pretty-printer similar to ``numpy.array2string`` (simplified).

        Time complexity: O(size)

        Space complexity: O(size + ndim)

        Returns: The string representing the :class:`~mininumpy.array.Array`.
        """
        return self._format_recursive(self._data, self.shape)

    # TODO: add unit test
    def reshape(self, newshape: tuple):
        """
        Reshape the array to the given dimensions if compatible.

        Args:
            newshape: Target shape.

        Time complexity: O(len(newshape))

        Space complexity: O(1)

        Returns: The reshaped array if valid.
        """
        if math.prod(newshape) != self.size:
            return
        return Array(self._data, shape=newshape)

    def __setitem__(self, coor, value):
        idx = shaped_to_flat_index(coor, self.shape)
        self._data[idx] = value

    def __getitem__(self, coor):
        idx = shaped_to_flat_index(coor, self.shape)
        return self._data[idx]

    def transpose(self, axis: list | tuple = None):
        """
        Return a new array with axes permuted.

        Args:
            axis: Axis order (e.g. transpose the last 2 axes with ``axis=(0, 1, 3, 2)`` for 4D array). If None, reverses the dimensions.

        Time complexity: O(size * ndim)

        Space complexity: O(size + ndim)

        Returns: A new array with transposed shape.
        """
        if axis == None:
            axis = range(self.ndim)[::-1]

        new_data = list(self._data)
        new_shape = list(self.shape)
        for i in range(len(axis)):
            new_shape[i] = self.shape[axis[i]]

        for i in range(self.size):
            new_data[transpose_new_index(i, self.shape, axis)] = self._data[i]

        return Array(data=new_data, shape=new_shape)

    def tolist(self):
        """
        Get the ``list`` representation of the :class:`~mininumpy.array.Array`. Similar to ``.data``

        Time complexity: O(size * ndim)

        Space complexity: O(size)

        Returns: The ``list`` representation of the :class:`~mininumpy.array.Array`
        """
        return self.data

    # TODO: add unit test
    def __add__(self, other):
        """
        Element-wise addition with another :class:`~mininumpy.array.Array` or a scalar.

        Args:
            other: Array of same shape or scalar to add.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: New :class:`~mininumpy.array.Array` with element-wise summation.
        """
        if isinstance(other, Array) and self.shape != other.shape:
            return
        ans = self._data.copy()
        if isinstance(other, Array):
            for i in range(self.size):
                ans[i] += other._data[i]
        else:
            for i in range(self.size):
                ans[i] += other
        return Array(ans, shape=self.shape)

    def __radd__(self, other):
        """
        Element-wise addition with another :class:`~mininumpy.array.Array` or a scalar.

        Args:
            other: Array of same shape or scalar to add.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: New :class:`~mininumpy.array.Array` with element-wise summation.
        """
        if other == 0:
            return self
        return self.__add__(other)

    # TODO: add unit test
    def __mul__(self, other):
        """
        Element-wise multiplication with another :class:`~mininumpy.array.Array` or a scalar.

        Args:
            other: Array of same shape or scalar to multiply.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: The Element-wise product :class:`~mininumpy.array.Array`.
        """
        if isinstance(other, Array) and self.shape != other.shape:
            return
        ans = self._data.copy()
        if isinstance(other, Array):
            for i in range(self.size):
                ans[i] *= other._data[i]
        else:
            for i in range(self.size):
                ans[i] *= other
        return Array(ans, shape=self.shape)

    def __rmul__(self, other):
        """
        Element-wise multiplication with another :class:`~mininumpy.array.Array` or a scalar.

        Args:
            other: Array of same shape or scalar to multiply.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: The Element-wise product :class:`~mininumpy.array.Array`.
        """
        return self.__mul__(other)

    def __sub__(self, other):
        """
        Element-wise subtraction with another :class:`~mininumpy.array.Array` or a scalar.

        Args:
            other: Array of same shape or scalar to subtract.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: New :class:`~mininumpy.array.Array` with element-wise subtraction.
        """
        return self + other * (-1)

    # TODO: add unit test
    def __pow__(self, other: int):
        """
        Element-wise exponentiation by a scalar.

        Args:
            other: Exponent to apply to each element.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: New array with elements raised to the given power.
        """
        ans = self._data.copy()
        for i in range(self.size):
            ans[i] **= other
        return Array(ans, shape=self.shape)

    def __truediv__(self, other):
        """
        Element-wise division by another :class:`~mininumpy.array.Array` or a scalar.

        Args:
            other: Divisor, either scalar or :class:`~mininumpy.array.Array` of same shape.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: :class:`~mininumpy.array.Array` of element-wise division.
        """
        return self.__mul__(other ** -1)

    def is_square(self):
        """
        Check whether the :class:`~mininumpy.array.Array` is a square matrix.

        Time complexity: O(1)

        Space complexity: O(1)

        Returns: ``True`` if the :class:`~mininumpy.array.Array` has 2 dimensions and both are equal, ``False`` otherwise.
        """
        if self.ndim != 2:
            return False
        return self.shape[0] == self.shape[1]

    def sum(self, axis: int = 0):
        """
        Compute the sum of array elements along a given axis.

        Args:
            axis: Axis along which to sum. Default ``axis`` is 0.

        Time complexity: O(size * ndim)

        Space complexity: O(size)

        Returns: :class:`~mininumpy.array.Array` of summed values along the specified axis.
        """
        result_shape = self.shape[:axis] + self.shape[axis+1:]
        result_size = 1
        for s in result_shape:
            result_size *= s

        ans = [0] * result_size

        for flat_index in range(self.size):

            # O(ndim)
            shaped_index = flat_index_to_shaped(flat_index, self.shape)

            reduced_index = shaped_index[:axis] + shaped_index[axis+1:]

            bucket = shaped_to_flat_index(reduced_index, result_shape)

            ans[bucket] += self._data[flat_index]

        return Array(ans, shape=result_shape)

    def mean(self, axis: int = 0):
        """
        Compute the average value of array elements along a given axis.

        Args:
            axis: Axis along which to take average. Default ``axis`` is 0.

        Time complexity: O(size * ndim)

        Space complexity: O(size)

        Returns: :class:`~mininumpy.array.Array` of averaged values along the specified axis.
        """
        return self.sum(axis) / self.shape[axis]

    # TODO: add `n` smallest
    def min(self):
        """
        Get the minimum element in the :class:`~mininumpy.array.Array`.

        Time complexity: O(size)

        Space complexity: O(1)

        Returns: The minimum element in the :class:`~mininumpy.array.Array`.
        """
        return min(self._data)

    # TODO: add `n` biggest
    def max(self):
        """
        Get the maximum element in the :class:`~mininumpy.array.Array`.

        Time complexity: O(size)

        Space complexity: O(1)

        Returns: The maximum element in the :class:`~mininumpy.array.Array`.
        """
        return max(self._data)

    # TODO: add `n` biggest
    def argmax(self) -> tuple[int]:
        """
        Get the index of the maximum element in the :class:`~mininumpy.array.Array`.

        Time complexity: O(size + ndim)

        Space complexity: O(ndim)

        Returns: The index of the maximum element in the :class:`~mininumpy.array.Array`.
        """
        max_idx = 0
        for i in range(self.size):
            if self._data[i] > self._data[max_idx]:
                max_idx = i
        return flat_index_to_shaped(max_idx)

    # TODO: add `n` smallest
    def argmin(self) -> tuple[int]:
        """
        Get the index of the minimum element in the :class:`~mininumpy.array.Array`.

        Time complexity: O(size + ndim)

        Space complexity: O(ndim)

        Returns: The index of the minimum element in the :class:`~mininumpy.array.Array`.
        """
        min_idx = 0
        for i in range(self.size):
            if self._data[i] < self._data[min_idx]:
                min_idx = i
        return flat_index_to_shaped(min_idx)

    def __matmul__(self, other):
        """
        Perform matrix multiplication (the ``@`` operator) between two :class:`~mininumpy.array.Array` objects.

        This supports batched matrix multiplication:
        - If ``self`` has shape (..., n, p) and ``other`` has shape (..., p, m),
            the result will have shape (..., n, m).
        - The leading dimensions (``...``) must match.

        Args:
            other: Right-hand side operand. Must be an :class:`~mininumpy.array.Array` with compatible dimensions (``self.shape[-1] == other.shape[-2]``).

        Time complexity: O(count * (ndim + n * m * p)) where count  = size / (n * p)

        Space complexity: O(n * m * count)

        Returns: New Array with shape (..., n, m) containing the matrix product. Returns None if ``other`` is not an Array or if dimensions are incompatible.
        """
        if not isinstance(other, Array):
            return
        if self.shape[-1] != other.shape[-2]:
            return

        count = math.prod(self.shape[:-2])

        _data_ans = []

        n = self.shape[-2]
        p = self.shape[-1]
        m = other.shape[-1]

        axes = list(range(other.ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]   # swap last two

        # O(count * m * p * ndim)
        otherT = other.transpose(axis=axes)

        # O(count * (ndim + n * m * p)) where count  = size / (n * p)
        for c in range(count):
            # O(ndim)
            start_self = c * n * p
            stop_self = start_self + n * p

            start_otherT = c * m * p
            stop_otherT = start_otherT + m * p

            # O(n * m * p)
            _data_ans += matmul_flat_2D(self._data[start_self:stop_self],
                                        otherT._data[start_otherT:stop_otherT],
                                        n, p, m)

        new_shape = list(self.shape)
        new_shape[-1] = m
        return Array(_data_ans, shape=tuple(new_shape))

    def elementwise(self, func):
        """
        Apply a function element-wise to all entries of the :class:`~mininumpy.array.Array`.

        Args:
            func: Function applied to each element.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: A new :class:`~mininumpy.array.Array` with the same shape as the input, where each element is the result of applying ``func`` to the corresponding input element.
        """
        ans = self._data.copy()
        for i in range(self.size):
            ans[i] = func(self._data[i])
        return Array(ans, shape=self.shape)

    def __exp__(self):
        """
        Apply the exponential function element-wise to an :class:`~mininumpy.array.Array`.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: New Array with elements transformed by ``exp(x)``.
        """
        return self.elementwise(math.exp)

    def __log__(self):
        """
        Apply the natural logarithm element-wise to the :class:`~mininumpy.array.Array`.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: New Array with elements transformed by ``log(x)``.
        """
        return self.elementwise(math.log)

    def __sqrt__(self):
        """
        Apply the square root function element-wise to the :class:`~mininumpy.array.Array`.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: New Array with elements transformed by ``sqrt(x)``.
        """
        return self.elementwise(math.sqrt)

    def __abs__(self):
        """
        Apply the absolute value function element-wise to the :class:`~mininumpy.array.Array`.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: New Array with elements transformed by ``abs(x)``.
        """
        return self.elementwise(abs)


def array(data: list) -> Array:
    """
    Create a new Array instance from a given (nested) list ``data``. This is a convenience wrapper around the Array constructor, similar to NumPy's ``np.array``.

    Args:
        data: Input (nested) list to be converted into an :class:`~mininumpy.array.Array`.

    Time complexity: O(size * ndim)

    Space complexity: O(size)

    Returns: A new :class:`~mininumpy.array.Array` instance wrapping the input data.
    """
    return Array(data)


def zeros(shape: tuple[int]) -> Array:
    """
    Create a new :class:`~mininumpy.array.Array` filled with zeros.

    Args:
        shape: Shape of the array to create.

    Time complexity: O(prod(shape))

    Space complexity: O(prod(shape))

    Returns:
        Array: An :class:`~mininumpy.array.Array` of the given ``shape`` filled with zeros.
    """
    return Array([0] * math.prod(shape), shape=shape)


def ones(shape: tuple[int]) -> Array:
    """
    Create a new :class:`~mininumpy.array.Array` filled with ones.

    Args:
        shape: Shape of the array to create.

    Time complexity: O(prod(shape))

    Space complexity: O(prod(shape))

    Returns:
        Array: An :class:`~mininumpy.array.Array` of the given ``shape`` filled with ones.
    """
    return Array([1] * math.prod(shape), shape=shape)


def eye(n: int) -> Array:
    """
    Create an identity matrix of size n * n.

    Args:
        n: Dimension of the square identity matrix.

    Time complexity: O(n ^ 2)

    Space complexity: O(n ^ 2)

    Returns: An (n * n) identity matrix.
    """
    iden = [[0] * n for _ in range(n)]
    for i in range(n):
        iden[i][i] = 1
    return Array(iden)


def arange(start: float | int, stop: float | int, step: float | int = 1):
    """
    Create an :class:`~mininumpy.array.Array` with evenly spaced values in [start, stop) with given step.

    Args:
        start: Starting value.
        stop: End value (exclusive).
        step: Step size. Default is 1.

    Time complexity: O((stop - start) / step)

    Space complexity: O((stop - start) / step)

    Returns: 1D :class:`~mininumpy.array.Array` of evenly spaced values.
    """
    arr = []
    for i in range(int((stop - start) / step)):
        arr.append(start + i * step)
    return Array(arr)


def linspace(start: float | int, stop: float | int, num: int = 50):
    """
    Create an :class:`~mininumpy.array.Array` of evenly spaced values between start and stop (inclusive).

    Args:
        start: Starting value.
        stop: End value (inclusive).
        num: Number of samples to generate. Default is 50.

    Time complexity: O(num)

    Space complexity: O(num)

    Returns: 1D :class:`~mininumpy.array.Array` of evenly spaced values.
    """
    step = (stop - start) / (num - 1)
    return arange(start, stop + step, step)
