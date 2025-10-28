import mininumpy as mnp
import numpy as np

sample_lst = [list(range(1_000_000)),
              [[[[[0, 1], [-100, 1]]]]],
              [[1, 2, 3], 
               [4, 5, 6], 
               [7, 0, 9]]
              ]

def _test_array(attr, params=None):
    for l in sample_lst:
        if params == None:
            a = getattr(mnp.array(l), attr)
            b = getattr(np.array(l), attr)
        else:
            a = getattr(mnp.array(l), attr)(*params)
            b = getattr(np.array(l), attr)(*params)
        if isinstance(a, mnp.Array) and isinstance(b, np.ndarray):
            a = a.tolist()
            b = b.tolist()
        if a != b:
            return False
    return True


def test_array_shape():
    assert _test_array('shape')

def test_array_ndim():
    assert _test_array('ndim')

def test_array_size():
    assert _test_array('size')

def test_array_size():
    assert _test_array('size')

def test_array_data():
    assert _test_array('tolist', params=())

def test_array_transpose():
    assert _test_array('transpose', params=())
    

