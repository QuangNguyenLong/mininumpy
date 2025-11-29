import mininumpy as mnp
import numpy as np
import random

N_minor = [2, 3, 4, 5]
N_small = [20, 30, 40, 50]
N_large = [200, 300, 400, 500]
N_real = [2000, 3000, 4000, 5000]

sample_square = [[[[random.random() for _ in range(n)] for _ in range(n)] for n in N_minor],
                 [[[random.random() for _ in range(n)] for _ in range(n)]
                  for n in N_small],
                 [[[random.random() for _ in range(n)] for _ in range(n)]
                  for n in N_large],
                 [[[random.random() for _ in range(n)] for _ in range(n)] for n in N_real]]

zero_division_check = [[[1, 2, 3],
                       [0, 2, -34],
                       [-7, 0, 1]],
                       [[1/260,	-1/260,	-37/260],
                        [119/260,	11/260,	 17/260],
                        [7/260,	-7/260,	  1/260]]]


def test_array_inv():
    for sample in sample_square[0] + zero_division_check:
        a = mnp.minilinalg.inv(mnp.array(sample)).tolist()
        b = np.linalg.inv(np.array(sample)).tolist()

        if not np.allclose(a, b):
            print("[TEST FAILED]\n mnp: ", a, "\nnp: ", b)
            assert False
        else:
            assert True


def test_array_inv_LU():
    for sample in sample_square[1] + zero_division_check:
        a = mnp.minilinalg.inverse_LU(mnp.array(sample)).tolist()
        b = np.linalg.inv(np.array(sample)).tolist()

        if not np.allclose(a, b):
            print("[TEST FAILED]\n mnp: ", a, "\nnp: ", b)
            assert False
        else:
            assert True


def test_array_det():
    for sample in sample_square[0] + zero_division_check:
        a = mnp.minilinalg.det(mnp.array(sample))
        b = np.linalg.det(np.array(sample))

        if not np.allclose(a, b):
            print("[TEST FAILED]\n mnp: ", a, "\nnp: ", b)
            assert False
        else:
            assert True


def test_array_det_LU():
    for sample in sample_square[1] + zero_division_check:
        a = mnp.minilinalg.det_LU(mnp.array(sample))
        b = np.linalg.det(np.array(sample))

        if not np.allclose(a, b):
            print("[TEST FAILED]\n mnp: ", a, "\nnp: ", b)
            assert False
        else:
            assert True

def test_array_det_gaussian():
    for sample in sample_square[1] + zero_division_check:
        a = mnp.minilinalg.det_gaussian(mnp.array(sample))
        b = np.linalg.det(np.array(sample))

        if not np.allclose(a, b):
            print("[TEST FAILED]\n mnp: ", a, "\nnp: ", b)
            assert False
        else:
            assert True