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

# nxp x pxm
# TODO: Optimize PLS


def matmul_flat_2D(A: list[float], BT: list[float], n: int, p: int, m: int) -> list[float]:
    """
    Perform 2D matrix multiplication in flattened form.

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
    Helper function to flat a nested list.

    Args:
        lst: nested list to be flatten.

    Time complexity: O(size * ndim)

    Space complexity: O(size)

    Returns: flatten list.
    """
    if not isinstance(lst, list):
        return lst

    flat = lst
    # O(size * ndim)
    while isinstance(flat[0], list):
        ans = []
        # O(size)
        for e in flat:
            # O(len(e))
            ans += e
        flat = ans

    lst = flat
    return lst


class Array:
    """
    Numpy-like array. Store the data in a flatten array.
    """

    _data = []
    """
    The raw flatten data of the ``Array``.
    """

    element_type = None
    """
    The type of elements stored in the ``Array``.
    """

    shape = ()
    """
    The dimensions of the ``Array``.
    """

    ndim = 0
    """
    The number of dimensions of the ``Array``.
    """

    size = 0
    """
    The total number of elements in the ``Array``.
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
                next.append(curr[i * c: i * c + c])
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

    def __init__(self, data: list, shape: tuple[int] = None, element_type = None):
        """
        Initialize an ``Array`` object from nested list ``data``.

        Args:
            data: Input nested list.
            shape: Desired shape. If ``None``, inferred automatically.
            element_type: Type to cast elements. If ``None``, inferred from data.

        Time complexity: O(size * ndim)

        Space complexity: O(size)

        Returns: None
        """
        if not isinstance(data, list):
            raise ValueError("wtf bro")

        if shape:
            # O(size * ndim)
            self._data = flatten(data)
            # O(ndim)
            size = math.prod(shape)
            if len(self._data) != size:
                raise ValueError(
                    f"Error: Can not change size. From {len(self._data)} to {size}.")
            self.shape = shape
        else:
            # O(ndim)
            self.shape = self._shape(data)
            # O(size * ndim)
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

        Returns: The string representing the ``Array``.
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
        Get the ``list`` representation of the ``Array``. Similar to ``.data``

        Time complexity: O(size * ndim)

        Space complexity: O(size)

        Returns: The ``list`` representation of the ``Array``
        """
        return self.data

    # TODO: add unit test
    def __add__(self, other):
        """
        Element-wise addition with another ``Array`` or a scalar.

        Args:
            other: Array of same shape or scalar to add.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: New ``Array`` with element-wise summation.
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
        Element-wise addition with another ``Array`` or a scalar.

        Args:
            other: Array of same shape or scalar to add.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: New ``Array`` with element-wise summation.
        """
        if other == 0:
            return self
        return self.__add__(other)

    # TODO: add unit test
    def __mul__(self, other):
        """
        Element-wise multiplication with another ``Array`` or a scalar.

        Args:
            other: Array of same shape or scalar to multiply.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: The Element-wise product ``Array``.
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
        Element-wise multiplication with another ``Array`` or a scalar.

        Args:
            other: Array of same shape or scalar to multiply.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: The Element-wise product ``Array``.
        """
        return self.__mul__(other)

    def __sub__(self, other):
        """
        Element-wise subtraction with another ``Array`` or a scalar.

        Args:
            other: Array of same shape or scalar to subtract.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: New ``Array`` with element-wise subtraction.
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
        Element-wise division by another ``Array`` or a scalar.

        Args:
            other: Divisor, either scalar or ``Array`` of same shape.

        Time complexity: O(size)

        Space complexity: O(size)

        Returns: ``Array`` of element-wise division.
        """
        return self.__mul__(other ** -1)

    def is_square(self):
        """
        Check whether the ``Array`` is a square matrix.

        Time complexity: O(1)

        Space complexity: O(1)

        Returns: ``True`` if the ``Array`` has 2 dimensions and both are equal, ``False`` otherwise.
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

        Returns: ``Array`` of summed values along the specified axis.
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

        Returns: ``Array`` of averaged values along the specified axis.
        """
        return self.sum(axis) / self.shape[axis]

    # TODO: add `n` smallest
    def min(self):
        """
        Get the minimum element in the ``Array``.

        Time complexity: O(size)

        Space complexity: O(1)

        Returns: The minimum element in the ``Array``.
        """
        return min(self._data)

    # TODO: add `n` biggest
    def max(self):
        """
        Get the maximum element in the ``Array``.

        Time complexity: O(size)

        Space complexity: O(1)

        Returns: The maximum element in the ``Array``.
        """
        return max(self._data)

    # TODO: add `n` biggest
    def argmax(self) -> tuple[int]:
        """
        Get the index of the maximum element in the ``Array``.

        Time complexity: O(size + ndim)

        Space complexity: O(ndim)

        Returns: The index of the maximum element in the ``Array``.
        """
        max_idx = 0
        for i in range(self.size):
            if self._data[i] > self._data[max_idx]:
                max_idx = i
        return flat_index_to_shaped(max_idx)

    # TODO: add `n` smallest
    def argmin(self) -> tuple[int]:
        """
        Get the index of the minimum element in the ``Array``.

        Time complexity: O(size + ndim)

        Space complexity: O(ndim)

        Returns: The index of the minimum element in the ``Array``.
        """
        min_idx = 0
        for i in range(self.size):
            if self._data[i] < self._data[min_idx]:
                min_idx = i
        return flat_index_to_shaped(min_idx)

    def __matmul__(self, other):
        """
        Perform matrix multiplication (the ``@`` operator) between two ``Array`` objects.

        This supports batched matrix multiplication:
        - If ``self`` has shape (..., n, p) and ``other`` has shape (..., p, m),
            the result will have shape (..., n, m).
        - The leading dimensions (``...``) must match.

        Args:
            other: Right-hand side operand. Must be an ``Array`` with compatible dimensions (``self.shape[-1] == other.shape[-2]``).

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
            shaped = flat_index_to_shaped(c, self.shape[:-2])

            start_self = shaped_to_flat_index(shaped + (0, 0), self.shape)
            stop_self = shaped_to_flat_index(shaped + (n-1, p-1), self.shape)

            start_otherT = shaped_to_flat_index(shaped + (0, 0), otherT.shape)
            stop_otherT = shaped_to_flat_index(
                shaped + (m-1, p-1), otherT.shape)

            # O(n * m * p)
            _data_ans += matmul_flat_2D(self._data[start_self:stop_self + 1],
                                        otherT._data[start_otherT:stop_otherT + 1],
                                        n, p, m)

        new_shape = list(self.shape)
        new_shape[-1] = m
        return Array(_data_ans, shape=tuple(new_shape))


def _elementwise(array: Array, func) -> Array:
    """
    Apply a function element-wise to all entries of an Array.

    Args:
        array: Input array to transform.
        func: Function applied to each element.

    Time complexity: O(array.size)

    Space complexity: O(array.size)

    Returns: A new ``Array`` with the same shape as the input, where each element is the result of applying ``func`` to the corresponding input element. Returns None if the input is not an Array.
    """
    if not isinstance(array, Array):
        return
    ans = array._data.copy()
    for i in range(array.size):
        ans[i] = func(array._data[i])
    return Array(ans, shape=array.shape)


def exp(array: Array) -> Array:
    """
    Apply the exponential function element-wise to an ``Array``.

    Args:
        array: Input array.

    Time complexity: O(size)

    Space complexity: O(size)

    Returns: New Array with elements transformed by ``exp(x)``.
    """
    return _elementwise(array, math.exp)


def log(array: Array) -> Array:
    """
    Apply the natural logarithm element-wise to an ``Array``.

    Args:
        array: Input array.

    Time complexity: O(size)

    Space complexity: O(size)

    Returns: New Array with elements transformed by ``log(x)``.
    """
    return _elementwise(array, math.log)


def sqrt(array: Array) -> Array:
    """
    Apply the square root function element-wise to an ``Array``.

    Args:
        array: Input array.

    Time complexity: O(size)

    Space complexity: O(size)

    Returns: New Array with elements transformed by ``sqrt(x)``.
    """
    return _elementwise(array, math.sqrt)


def abs(array: Array) -> Array:
    """
    Apply the absolute value function element-wise to an ``Array``.

    Args:
        array (Array): Input array.

    Time complexity: O(size)

    Space complexity: O(size)

    Returns: New Array with elements transformed by ``abs(x)``.
    """
    return _elementwise(array, math.fabs)


def array(data: list) -> Array:
    """
    Create a new Array instance from a given (nested) list ``data``. This is a convenience wrapper around the Array constructor, similar to NumPy's ``np.array``.

    Args:
        data: Input (nested) list to be converted into an ``Array``.

    Time complexity: O(size * ndim)

    Space complexity: O(size)

    Returns: A new ``Array`` instance wrapping the input data.
    """
    return Array(data)


def zeros(shape: tuple[int]) -> Array:
    """
    Create a new ``Array`` filled with zeros.

    Args:
        shape: Shape of the array to create.

    Time complexity: O(prod(shape))

    Space complexity: O(prod(shape))

    Returns:
        Array: An ``Array`` of the given ``shape`` filled with zeros.
    """
    return Array([0] * math.prod(shape), shape=shape)


def ones(shape: tuple[int]) -> Array:
    """
    Create a new ``Array`` filled with ones.

    Args:
        shape: Shape of the array to create.

    Time complexity: O(prod(shape))

    Space complexity: O(prod(shape))

    Returns:
        Array: An ``Array`` of the given ``shape`` filled with ones.
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
    Create an ``Array`` with evenly spaced values in [start, stop) with given step.

    Args:
        start: Starting value.
        stop: End value (exclusive).
        step: Step size. Default is 1.

    Time complexity: O((stop - start) / step)

    Space complexity: O((stop - start) / step)

    Returns: 1D ``Array`` of evenly spaced values.
    """
    arr = []
    for i in range(int((stop - start) / step)):
        arr.append(start + i * step)
    return Array(arr)


def linspace(start: float | int, stop: float | int, num: int = 50):
    """
    Create an ``Array`` of evenly spaced values between start and stop (inclusive).

    Args:
        start: Starting value.
        stop: End value (inclusive).
        num: Number of samples to generate. Default is 50.

    Time complexity: O(num)

    Space complexity: O(num)

    Returns: 1D ``Array`` of evenly spaced values.
    """
    step = (stop - start) / (num - 1)
    return arange(start, stop + step, step)
