# ---------------------------PART 1----------------------------------#
# this class is not meant for logging and printing :v
import math

# ----------------------------my stuff---------------------------------#


def dot(v1: list[float], v2: list[float]) -> float:
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
    ans = [0] * (m * n)
    for i in range(n):
        for j in range(m):
            ans[shaped_to_flat_index((i, j), (n, m))] = dot(
                A[i * p:(i+1) * p], BT[j * p: (j+1) * p])
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
    TODO: Add description.

    Args:
        vec:
        shape:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    ndim = len(shape)
    ans = 0
    for i in range(ndim):
        prod = 1
        for j in range(i + 1, ndim):
            prod *= shape[j]
        ans += prod * vec[i]

    return ans


def flat_index_to_shaped(index: int, shape: tuple[int]) -> tuple[int]:
    """
    TODO: Add description.

    Args:
        index:
        shape:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    ndim = len(shape)
    ans = [0] * ndim

    for i in range(ndim):
        prod = 1
        for j in range(i + 1, ndim):
            prod *= shape[j]
        ans[i] = int(index / prod)
        index -= int(index / prod) * prod

    return tuple(ans)

# TODO: add unit test


def flatten(lst: list) -> list:
    """
    TODO: Helper function to flat a nested list.

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
    element_type = None
    shape = ()
    ndim = 0
    size = 0

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
        Method to get the number of elements in the array.

        Time complexity: O(1)

        Space complexity: O(1)

        Returns: The number of elements in integer.
        """
        return len(self._data)

    def _shape(self) -> tuple[int]:
        """
        TODO: Add description.

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        shape = (len(self._data), )
        e = self._data[0]
        while isinstance(e, list):
            shape += (len(e), )
            e = e[0]
        return shape

    def _ndim(self):
        """
        TODO: Add description.

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        return len(self.shape)

    def __init__(self, lst, shape=None, element_type=None):
        """
        TODO: Add description.

        Args:
            lst:
            shape:
            element_type:

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        if not isinstance(lst, list):
            raise ValueError("wtf bro")

        self._data = lst
        if shape:
            self._data = flatten(lst)
            size = 1
            for s in shape:
                size *= s
            if len(self._data) != size:
                raise ValueError(
                    f"Error: Can not change size. From {len(self._data)} to {size}.")
            self.shape = shape
        else:
            self.shape = self._shape()
            self._data = flatten(lst)

        self.ndim = self._ndim()
        self.size = self._size()

        if element_type:
            self.element_type = element_type
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
        TODO: Add description.

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        return self._format_recursive(self._data, self.shape)

    # TODO: add unit test
    def reshape(self, newshape):
        """
        TODO: Add description.

        Args:
            newshape:

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        size = 1
        for c in newshape:
            size *= c
        if size != self.size:
            return 1

        self.shape = newshape
        self.ndim = self._ndim()
        return self

    def __setitem__(self, coor, value):
        idx = shaped_to_flat_index(coor, self.shape)
        self._data[idx] = value

    def __getitem__(self, coor):
        idx = shaped_to_flat_index(coor, self.shape)
        return self._data[idx]

    def transpose(self, axis=None):
        """
        TODO: Add description.

        Args:
            axis:

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        if axis == None:
            axis = range(self.ndim)[::-1]

        new_data = list(self._data)
        new_shape = list(self.shape)
        for i in range(len(axis)):
            new_shape[i] = self.shape[axis[i]]

        for i in range(self.size):
            new_data[transpose_new_index(i, self.shape, axis)] = self._data[i]

        return Array(lst=new_data, shape=new_shape)

    def tolist(self):
        """
        TODO: Add description.

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        return self.data

    # TODO: add unit test
    def __add__(self, other):
        """
        TODO: Add description.

        Args:
            other:

        Time complexity: O()

        Space complexity: O()

        Returns:
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
        TODO: Add description.

        Args:
            other:

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        if other == 0:
            return self
        return self.__add__(other)

    # TODO: add unit test
    def __mul__(self, other):
        """
        TODO: Add description.

        Args:
            other:

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        if isinstance(other, Array) and self.shape != other.shape:
            return
        ans = self
        if isinstance(other, Array):
            for i in range(self.size):
                ans._data[i] *= other._data[i]
        else:
            for i in range(self.size):
                ans._data[i] *= other
        return ans

    def __rmul__(self, other):
        """
        TODO: Add description.

        Args:
            other:

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        return self.__mul__(other)

    def __sub__(self, other):
        """
        TODO: Add description.

        Args:
            other:

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        return self + other * (-1)

    # TODO: add unit test
    def __pow__(self, other):
        """
        TODO: Add description.

        Args:
            other:

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        ans = self
        for i in range(self.size):
            ans._data[i] **= other
        return ans

    def __truediv__(self, other):
        """
        TODO: Add description.

        Args:
            other:

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        return self.__mul__(other ** -1)

    def is_square(self):
        """
        TODO: Add description.

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        if self.ndim != 2:
            return False
        return self.shape[0] == self.shape[1]

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True):
        """
        TODO: Add description.

        Time complexity: O()

        Space complexity: O()

        Returns:
        """

        tmp = self.transpose(
            (axis,) + tuple(i for i in range(self.ndim) if i != axis))

        element_shape = tmp.shape[1:]
        element_size = int(tmp.size / tmp.shape[0])
        ans = zeros(element_shape)

        for i in range(tmp.shape[0]):
            element_ith = tmp._data[i * element_size: (i + 1) * element_size]
            ans += Array(element_ith, element_shape)
        return ans

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        """
        TODO: Add description.

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        return self.sum(axis) / self.shape[axis]

    # TODO: add `n` smallest
    def min(self):
        """
        TODO: Add description.

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        return min(self._data)

    # TODO: add `n` biggest
    def max(self):
        """
        TODO: Add description.

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        return max(self._data)

    # TODO: add `n` biggest
    def argmax(self):
        """
        TODO: Add description.

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        max_idx = 0
        for i in range(self.size):
            if self._data[i] > self._data[max_idx]:
                max_idx = i
        return max_idx

    # TODO: add `n` smallest
    def argmin(self):
        """
        TODO: Add description.

        Time complexity: O()

        Space complexity: O()

        Returns:
        """
        max_idx = 0
        for i in range(self.size):
            if self._data[i] < self._data[max_idx]:
                max_idx = i
        return max_idx

    def __matmul__(self, other):
        """
        TODO: Add description.

        Args:
            other:

        Time complexity: O()

        Space complexity: O()

        Returns:
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
        otherT = other.transpose(axis=axes)

        for c in range(count):
            shaped = flat_index_to_shaped(c, self.shape[:-2])

            start_self = shaped_to_flat_index(shaped + (0, 0), self.shape)
            stop_self = shaped_to_flat_index(shaped + (n-1, p-1), self.shape)

            start_otherT = shaped_to_flat_index(shaped + (0, 0), otherT.shape)
            stop_otherT = shaped_to_flat_index(
                shaped + (m-1, p-1), otherT.shape)

            _data_ans += matmul_flat_2D(self._data[start_self:stop_self + 1],
                                        otherT._data[start_otherT:stop_otherT + 1],
                                        n, p, m)

        new_shape = list(self.shape)
        new_shape[-1] = m
        return Array(_data_ans, shape=tuple(new_shape))


def _elementwise(array, func):
    """
    TODO: Add description.

    Args:
        array:
        func:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    if not isinstance(array, Array):
        return
    ans = array
    for i in range(array.size):
        ans._data[i] = func(array._data[i])
    return ans


def exp(array):
    """
    TODO: Add description.

    Args:
        array:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    return _elementwise(array, math.exp)


def log(array):
    """
    TODO: Add description.

    Args:
        array:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    return _elementwise(array, math.log)


def sqrt(array):
    """
    TODO: Add description.

    Args:
        array:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    return _elementwise(array, math.sqrt)


def abs(array):
    """
    TODO: Add description.

    Args:
        array:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    return _elementwise(array, math.fabs)


def array(object):
    """
    TODO: Add description.

    Args:
        object:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    return Array(object)


def zeros(shape):
    """
    TODO: Add description.

    Args:
        shape:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    size = 1
    for s in shape:
        size *= s
    zero = [0] * size
    return Array(zero, shape=shape)


def ones(shape):
    """
    TODO: Add description.

    Args:
        shape:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    size = 1
    for s in shape:
        size *= s
    one = [1] * size
    return Array(one, shape=shape)


def eye(n):
    """
    TODO: Add description.

    Args:
        n:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    iden = [[0] * n for _ in range(n)]
    for i in range(n):
        iden[i][i] = 1
    return Array(iden)


def arange(start, stop, step=1):
    """
    TODO: Add description.

    Args:
        start:
        stop:
        step:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    arr = []
    for i in range(int((stop - start) / step)):
        arr.append(start + i * step)
    return Array(arr)


def linspace(start, stop, num=50):
    """
    TODO: Add description.

    Args:
        start:
        stop:
        num:

    Time complexity: O()

    Space complexity: O()

    Returns:
    """
    step = (stop - start) / (num - 1)
    return arange(start, stop + step, step)
