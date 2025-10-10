import mininumpy as mnp

points = mnp.array([
    [1, 1],
    [2, 3],
    [4, 2],
    [3, 5],
    [5, 4]
])

scale = mnp.array([[2, 0],
                   [0, 0.5]])

rotate = mnp.array([[0, -1],
                    [1, 0]])

a = 2
b = 0.5
shear = mnp.array([[1 + a * b, a],
                   [b, 1]])

# -------------scale-------------#

print(points @ scale)

# -------------rotate-------------#

print(points @ rotate)

# -------------shear-------------#

print(points @ shear)

print(mnp.zeros((3, 2)))
