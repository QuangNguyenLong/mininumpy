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
    TODO: Add description.

    Args:
        lst:

    Returns:
    """
    if not isinstance(lst, list):
        return lst

    flat = lst
    while isinstance(flat[0], list):
        ans = []
        for e in flat:
            ans += e
        flat = ans

    lst = flat
    return lst


class Array:
    """
    TODO: Add class description.
    """

    _data = []
    element_type = None
    shape = ()
    ndim = 0
    size = 0

    @property
    def data(self):
        """
        TODO: Add description.

        Returns:
        """
        curr = self._data
        for c in self.shape[:0:-1]:
            next = []
            for i in range(int(len(curr) / c)):
                next.append(curr[i * c: i * c + c])
            curr = next
        return curr

    def _size(self):
        """
        TODO: Add description.

        Returns:
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
            new_data[i] = self._data[transpose_new_index(i, self.shape, axis)]

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
        ans = self
        if isinstance(other, Array):
            for i in range(self.size):
                ans._data[i] += other._data[i]
        else:
            for i in range(self.size):
                ans._data[i] += other
        return ans

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


    def __getitem__(self, idx):
        """
        TODO: Add description.

        Args:
            idx:

        Returns:
        """
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.shape[0])
            ans = []
            if isinstance(self[0], Array):
                for i in range(start, stop, step):
                    ans.append(self[i]._data)
            else:
                ans = self._data[idx]
            sh = list(self.shape)
            sh[0] = int((stop - start) / step)

            return Array(ans, shape=tuple(sh))

        if self.ndim == 1:
            return self._data[idx]
        if idx >= self.shape[0]:
            raise IndexError("Array index out of range!!!")

        block = int(self.size / self.shape[0])
        return Array(self._data[block * idx: block * (idx + 1)],
                     shape=self.shape[1:])
    
    def sum(self, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True):
        """
        TODO: Add description.
        """
        return

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        """
        TODO: Add description.
        """
        return sum(self._data) / self.size

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
                row.append(sum(self[i] * otherT[j]))
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
