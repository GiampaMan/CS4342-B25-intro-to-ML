import numpy as np

def problem1 (A, B, C):
    return (A @ B) - C

def problem2 (A):
    return np.ones(A.size)

def problem3 (A):
    return np.fill_diagonal(A, 0)

def problem4 (A, i):
    return np.sum(A[i])

def problem5 (A, c, d):
    bounds = (A >= c) & (A <= d)
    return np.mean(A[bounds])

def problem6 (A, k):
    eigvals, eigvecs = np.linalg.eig(A)

    magnitudes = np.abs(eigvals)

    top_indices = np.argsort(magnitudes)[::-1][:k]

    top_eigvecs = eigvecs[:, top_indices]

    return top_eigvecs

def problem7 (A, x):
    return np.linalg.solve(A, x)

def problem8 (x, k):
     return np.repeat(np.atleast_2d(x).T, k, axis=1)

def problem9 (A):
    perm = np.random.permutation(A.shape[0])
    return A[perm, :]

def problem10 (A):
    return A.np.mean(axis=1)

def problem11 (n, k):
    arr = np.random.randint(0, k + 1, size=n)
    for i in arr:    
        if arr[i]% 2 == 0:
            arr[i] = -1
    return arr

# might not be correct
def problem12 (A, b):
    b = np.asarray(b).reshape(-1, 1)
    return A + b

def problem13 (A):
    n = A.shape[0]
    return A.reshape(n, -1).T
