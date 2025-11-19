from mininumpy.array import *
import cmath
def dot(a, b):
    return a @ b

def matmul(a, b):
    return a @ b

def norm(a):
    sm = sum(a)
    while isinstance(sm, Array):
        sm = sum(sm)
    return a / sm

def det(a):
    if not isinstance(a, Array):
        return
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return
    n = a.shape[0]
    if n == 1:
        return a._data[0]
    
    data = a.data

    ans = 0
    for i in range(n):
        cooef = Array([
            data[x][y]
            for x in range(n)
            for y in range(1, n)
            if x != i
        ], shape = (n - 1, n - 1))
        ans += (-1)**i * data[i][0] * det(cooef)
    return ans

def inv(a):
    if not isinstance(a, Array):
        return
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return
    n = a.shape[0]

    cofactor = Array([            
        (-1) ** (i + j) * det(Array([
            a[x][y]
            for x in range(n)
            for y in range(n)
            if x != i and y != j
        ], shape = (n - 1, n - 1)))
        for i in range(n)
        for j in range(n)], shape = a.shape)
    
    return cofactor.transpose() / det(a)

def eig(a):
    if not isinstance(a, Array):
        return
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return
    n = a.shape[0]

    eig = [i for i in range(n)]  # making sure they start differently

    for _ in range(100):
        
        prod = [
            math.prod(eig[i] - eig[j] for j in range(n) if i != j)
            for i in range(n)]
        
        f = [det(a - eye(n) * eig[i]) for i in range(n)]
    
        for i in range(n):
            eig[i] -= f[i] / prod[i]

    return eig