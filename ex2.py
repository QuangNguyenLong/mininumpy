from mininumpy.array import *
from mininumpy.minilinalg import *

points = Array([
    [1, 1],
    [2, 3],
    [4, 2],
    [3, 5],
    [5, 4]
])

scale = Array([[2, 0],
               [0, 0.5]])

rotate = Array([[0, -1],
                [1, 0]])

a = 2
b = 0.5
shear = Array([[1 + a * b, a],
               [b, 1]])

#-------------scale-------------#

print(points @ scale)

#-------------rotate-------------#

print(points @ rotate)

#-------------shear-------------#

print(points @ shear)