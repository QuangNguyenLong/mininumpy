# ---------------------------PART 1----------------------------------#
# this class is not meant for logging and printing :v
import math

# ----------------------------my stuff---------------------------------#

def transpose_new_index(index, shape, axis):
    """
    TODO: Add description.

    Args:
        index:
        shape:
        axis:

    Returns:
    """
    vec = flat_index_to_shaped(index, shape)

    new_vec = list(vec)
    new_shape = list(shape)
    for i in range(len(axis)):
        new_vec[i] = vec[axis[i]]
        new_shape[i] = shape[axis[i]] 

    return shaped_to_flat_index(new_vec, new_shape)


def shaped_to_flat_index(vec, shape):
    """
    TODO: Add description.

    Args:
        vec:
        shape:

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


def flat_index_to_shaped(index, shape):
    """
    TODO: Add description.

    Args:
        index:
        shape:

    Returns:
    """
    ndim = len(shape)
    ans = [0 for _ in range(ndim)]
    
    for i in range(ndim):
        prod = 1
        for j in range(i + 1, ndim):
            prod *= shape[j]
        ans[i] = int(index / prod)
        index -= int(index / prod) * prod 
    
    return ans


def flatten(lst):
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
    def data(self):
        """
        Method to get the original data.

        Time complexity: O(size * ndim)

        Space complexity: O(size)

        Returns: The original data in a (nested) list.
        """
        curr = self._data
        for c in self.shape[:0:-1]: # ndim
            next = []
            for i in range(int(len(curr) / c)): # size / shape[i] = size
                next.append(curr[i * c: i * c + c])
            curr = next
        return curr

    def _size(self):
        """
        Method to get the number of elements in the array.

        Time complexity: O(1)

        Space complexity: O(1)
        
        Returns: The number of elements in integer.
        """
        return len(self._data)

    def _shape(self):
        """
        TODO: Add description.

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

    def __str__(self):
        """
        TODO: Add description.

        Returns:
        """
        return self.data.__str__()

    def reshape(self, newshape):
        """
        TODO: Add description.

        Args:
            newshape:

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

    def transpose(self, axis=None):
        """
        TODO: Add description.

        Args:
            axis:

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

        Returns:
        """
        return self.data

    def __add__(self, other):
        """
        TODO: Add description.

        Args:
            other:

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

        Returns:
        """
        if other == 0:
            return self
        return self.__add__(other)

    def __mul__(self, other):
        """
        TODO: Add description.

        Args:
            other:

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

        Returns:
        """
        return self.__mul__(other)

    def __sub__(self, other):
        """
        TODO: Add description.

        Args:
            other:

        Returns:
        """
        return self + other * (-1)

    def __pow__(self, other):
        """
        TODO: Add description.

        Args:
            other:

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

        Returns:
        """
        return self.__mul__(other ** -1)

    
    def sum(self, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True):
        """
        TODO: Add description.
        """
        
        tmp = self.transpose((axis,) + tuple(i for i in range(self.ndim) if i != axis))

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
        """

        return self.sum(axis) / self.shape[axis]

    def min(self):
        """
        TODO: Add description.
        """
        return
    
    def max(self):
        """
        TODO: Add description.
        """
        return

    def argmax(self):
        """
        TODO: Add description.

        Returns:
        """
        max_idx = 0
        for i in range(self.size):
            if self._data[i] > self._data[max_idx]:
                max_idx = i
        return max_idx

    def argmin(self):
        """
        TODO: Add description.

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

        Returns:
        """
        if not isinstance(other, Array):
            return
        if self.shape[-1] != other.shape[0]:
            return
        if self.ndim > 2 or other.ndim > 2:
            return

        otherT = other.transpose()
        ans = []
        for i in range(self.shape[0]):
            row = []
            for j in range(otherT.shape[0]):
                row += (Array(self.data[i]) * Array(otherT.data[j])).sum(0).data
            ans.append(row)
        return Array(ans)


def _elementwise(array, func):
    """
    TODO: Add description.

    Args:
        array:
        func:

    Returns:
    """
    ans = array
    for i in range(array.size):
        ans._data[i] = func(array._data[i])
    return ans


def exp(array):
    """
    TODO: Add description.

    Args:
        array:

    Returns:
    """
    return _elementwise(array, math.exp)


def log(array):
    """
    TODO: Add description.

    Args:
        array:

    Returns:
    """
    return _elementwise(array, math.log)


def sqrt(array):
    """
    TODO: Add description.

    Args:
        array:

    Returns:
    """
    return _elementwise(array, math.sqrt)


def abs(array):
    """
    TODO: Add description.

    Args:
        array:

    Returns:
    """
    return _elementwise(array, math.fabs)


def array(object):
    """
    TODO: Add description.

    Args:
        object:

    Returns:
    """
    return Array(object)


def zeros(shape):
    """
    TODO: Add description.

    Args:
        shape:

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

    Returns:
    """
    step = (stop - start) / (num - 1)
    return arange(start, stop + step, step)
