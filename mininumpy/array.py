# ---------------------------PART 1----------------------------------#
# this class is not meant for logging and printing :v
import math

# ----------------------------my stuff---------------------------------#

def transpose_new_index(index, shape, axis):
    """
    @brief 
    @param 
    @param 
    @param 
    @return 
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
    @brief 
    @param 
    @param 
    @return 
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
    @brief 
    @param index
    @param shape
    @return 
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

# O(size * ndim) --> need improvement

def flatten(lst):
    """
    @brief 
    @param 
    @return 
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
    _data = []

    element_type = None

    shape = ()
    ndim = 0
    size = 0

    # ----------------------------end---------------------------------#

    # O(size * ndim)
    @property
    def data(self):
        # O(size * ndim)
        curr = self._data
        # O(product(shape) * ndim) = O(size * ndim)
        for c in self.shape[:0:-1]:
            next = []
            # O(product(shape)) = O(size)
            for i in range(int(len(curr) / c)):
                next.append(curr[i * c: i * c + c])
            curr = next
        return curr

    # O(1)
    def _size(self):
        return len(self._data)

    # O(ndim)
    def _shape(self):
        # O(1)
        shape = (len(self._data), )
        e = self._data[0]

        # O(ndim)
        while isinstance(e, list):
            # O(1)
            shape += (len(e), )
            e = e[0]
        return shape

    # O(1)
    def _ndim(self):
        return len(self.shape)

    # O(ndim * size), O(1) if list is flat
    def __init__(self, lst, shape=None, element_type=None):
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

    # O(ndim * size)
    def __str__(self):
        return self.data.__str__()

    # O(ndim)
    def reshape(self, newshape):
        # O(ndim)
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
        https://numpy.org/doc/2.3/reference/generated/numpy.ndarray.transpose.html
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
         return self.data
# ---------------------------PART 3----------------------------------#

    def __add__(self, other):
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
        if other == 0:
            return self
        return self.__add__(other)

    def __mul__(self, other):
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
        return self.__mul__(other)

    def __sub__(self, other):
        return self + other * (-1)

    def __truediv__(self, other):
        if isinstance(other, Array) and self.shape != other.shape:
            return
        ans = self
        if isinstance(other, Array):
            for i in range(self.size):
                ans._data[i] /= other._data[i]
        else:
            for i in range(self.size):
                ans._data[i] /= other
        return ans

    def __pow__(self, other):
        ans = self
        for i in range(self.size):
            ans._data[i] **= other
        return ans

    # O(1) if idx is int, but this is stupid because it is returning an entire new Array.
    def __getitem__(self, idx):
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
        return

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        return sum(self._data) / self.size

    def min(self):
        return
    
    def max(self):
        return

    def argmax(self):
        max_idx = 0
        for i in range(self.size):
            if self._data[i] > self._data[max_idx]:
                max_idx = i
        return max_idx

    def argmin(self):
        max_idx = 0
        for i in range(self.size):
            if self._data[i] < self._data[max_idx]:
                max_idx = i
        return max_idx

    def __matmul__(self, other):
        if not isinstance(other, Array):
            return
        if self.shape[-1] != other.shape[0]:
            return
        if self.ndim > 2 or other.ndim > 2:
            return  # this is confusing, pass :v

        otherT = other.transpose()
        ans = []
        for i in range(self.shape[0]):
            row = []
            for j in range(otherT.shape[0]):
                row.append(sum(self[i] * otherT[j]))
            ans.append(row)
        return Array(ans)

    

def _elementwise(array, func):
    ans = array
    for i in range(array.size):
        ans._data[i] = func(array._data[i])
    return ans


def exp(array):
    return _elementwise(array, math.exp)


def log(array):
    return _elementwise(array, math.log)


def sqrt(array):
    return _elementwise(array, math.sqrt)


def abs(array):
    return _elementwise(array, math.fabs)


# ---------------------------PART 2----------------------------------#


def array(object):
    return Array(object)

# O(len(shape) + prod(shape))


def zeros(shape):
    size = 1
    for s in shape:
        size *= s
    zero = [0] * size
    return Array(zero, shape=shape)

# O(len(shape) + prod(shape))


def ones(shape):
    size = 1
    for s in shape:
        size *= s
    one = [1] * size
    return Array(one, shape=shape)


def eye(n):
    iden = [[0] * n for _ in range(n)]
    for i in range(n):
        iden[i][i] = 1
    return Array(iden)


def arange(start, stop, step=1):
    arr = []
    for i in range(int((stop - start) / step)):
        arr.append(start + i * step)
    return Array(arr)


def linspace(start, stop, num=50):
    step = (stop - start) / (num - 1)
    return arange(start, stop + step, step)
