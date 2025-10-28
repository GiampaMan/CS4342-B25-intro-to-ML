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
    # If no entries match, np.mean on empty slice returns NaN, which is acceptable unless specified otherwise.
    return np.mean(A[bounds])

def problem6 (A, k):
    return ...

def problem7 (A, x):
    return ...

def problem8 (x, k):
    return ...

def problem9 (A):
    return ...

def problem10 (A):
    return ...

def problem11 (n, k):
    return ...

def problem12 (A, b):
    return ...

def problem13 (A):
    return ...
