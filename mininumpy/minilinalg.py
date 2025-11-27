from mininumpy.array import *
def dot(a, b):
    return a @ b

def matmul(a, b):
    return a @ b

def norm(a):
    sm = sum(a)
    while isinstance(sm, Array):
        sm = sum(sm)
    return a / sm

def LU_decomposition(A):
        if not isinstance(A, Array):
            return
        
        if A.ndim != 2:
            return

        L = [[1 if i == j else 0 for j in range(A.shape[0])] for i in range(A.shape[0])]

        U = []
        for i in range(A.shape[0]):
            row = []
            for j in range(A.shape[1]):
                row.append(A.element_type(A.data[i][j]))
            U.append(row)

        for i in range(A.shape[0] - 1):
            for j in range(i + 1, A.shape[0]):
                L[j][i] = U[j][i] / U[i][i]
                for k in range(A.shape[1]):
                    U[j][k] -= U[i][k] * L[j][i]
    
        return (L, U)

# TODO: Optimize PLS
# O(n!)
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

def det_LU(A):
    _, U = LU_decomposition(A)
    det = 1
    for i in range(len(U)):
        det *= U[i][i]
    return det

def forward_sub_inverse_LU(A, L, b):
    n = len(L)
    y = [0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    return y

def backward_sub_inverse_LU(A, U, y):
    n = len(U)
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]
    return x

def inverse_LU(A):
    if not isinstance(A, Array):
        return
    if not A.is_square():
        return
    
    identity = [[1 if i == j else 0 for j in range(A.shape[0])] for i in range(A.shape[0])]
    ans = []
    L, U = LU_decomposition(A)
    for i in range(A.shape[0]):
        ans.append(A.backward_sub_inverse_LU(U, A.forward_sub_inverse_LU(L, identity[i])))
    return Array(ans).transpose()

# TODO: Optimize PLS
# O(n!)
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

# TODO: complete eig value and VECTOR PLS, change algo to O(n^3) using TP2
def eig(a, max_iter=100, eps=1e-10):
    if not isinstance(a, Array):
        return
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return
    n = a.shape[0]

    # O(n)
    eig = [i for i in range(n)]  # making sure they start differently

    for _ in range(max_iter):
        #O(n^2)
        prod = [
            math.prod(eig[i] - eig[j] for j in range(n) if i != j)
            for i in range(n)]
        
        # O(n^4)
        f = [det(a - eye(n) * eig[i]) for i in range(n)]
    
        for i in range(n):
            eig[i] -= f[i] / prod[i]

    return eig

# TODO: 