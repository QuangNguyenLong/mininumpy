import mininumpy as mnp
import numpy as np
import random

sample_lst = [[[random.random() for _ in range(7)] for _ in range(123)],
              [[random.random() for _ in range(6)] for _ in range(7)],
              [[[random.random() for _ in range(2)] for _ in range(5)]
               for _ in range(8)],
              [[1, 2, 3],
               [4, 5, 6],
               [7, 0, 9]]
              ]


def test_array_shape():
    for sample in sample_lst:
        a = mnp.array(sample).shape
        b = np.array(sample).shape
        if not np.allclose(a, b):
            print("[TEST FAILED]\n mnp: ", a, "\nnp: ", b)
            assert False
        else:
            assert True


def test_array_ndim():
    for sample in sample_lst:
        a = mnp.array(sample).ndim
        b = np.array(sample).ndim
        if not np.allclose(a, b):
            print("[TEST FAILED]\n mnp: ", a, "\nnp: ", b)
            assert False
        else:
            assert True


def test_array_size():
    for sample in sample_lst:
        a = mnp.array(sample).size
        b = np.array(sample).size
        if not np.allclose(a, b):
            print("[TEST FAILED]\n mnp: ", a, "\nnp: ", b)

            assert False
        else:
            assert True


def test_array_data():
    for sample in sample_lst:
        a = mnp.array(sample).tolist()
        b = np.array(sample).tolist()
        if not np.allclose(a, b):
            print("[TEST FAILED]\n mnp: ", a, "\nnp: ", b)

            assert False
        else:
            assert True


def test_array_transpose():
    for sample in sample_lst:
        a = mnp.array(sample).transpose().tolist()
        b = np.array(sample).transpose().tolist()
        if not np.allclose(a, b):
            print("[TEST FAILED]\n mnp: ", a, "\nnp: ", b)

            assert False
        else:
            assert True


def test_array_sum():
    for sample in sample_lst:
        a = mnp.array(sample).sum(0).tolist()
        b = np.array(sample).sum(0).tolist()
        if not np.allclose(a, b):
            print("[TEST FAILED]\n mnp: ", a, "\nnp: ", b)
            assert False
        else:
            assert True


def test_array_mean():
    for sample in sample_lst:
        a = mnp.array(sample).mean(0).tolist()
        b = np.array(sample).mean(0).tolist()
        if not np.allclose(a, b):
            print("[TEST FAILED]\n mnp: ", a, "\nnp: ", b)
            assert False
        else:
            assert True


def test_array_matmul():
    a1 = mnp.array(sample_lst[0])
    a2 = mnp.array(sample_lst[1])
    a = (a1 @ a2).tolist()
    b1 = np.array(sample_lst[0])
    b2 = np.array(sample_lst[1])
    b = (b1 @ b2).tolist()
    if not np.allclose(a, b):
        print("[TEST FAILED]\n mnp: ", a, "\nnp: ", b)
        assert False
    else:
        assert True
