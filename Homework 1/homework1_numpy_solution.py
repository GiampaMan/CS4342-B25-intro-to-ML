import numpy as np

def problem1(A, B, C):
    """
    Given matrices A, B, and C, return A @ B - C.
    """
    return A @ B - C

def problem2(A):
    """
    Given matrix A, return a vector (1-D) of length equal to the number of rows of A,
    filled with ones.
    """
    return np.ones(A.shape[0])

def problem3(A):
    """
    Given matrix A, return a matrix with the same shape and contents as A except
    the diagonal entries are set to zero. Does not modify A in-place.
    """
    out = A.copy()
    # Works for non-square too; only sets the main diagonal indices that exist.
    np.fill_diagonal(out, 0)
    return out

def problem4(A, i):
    """
    Given matrix A and integer i, return the sum of the entries in the i-th row.
    No loops.
    """
    return A[i].sum()

def problem5(A, c, d):
    """
    Given matrix A and scalars c, d, compute the arithmetic mean over all entries of A
    that lie in [c, d], inclusive.
    """
    mask = (A >= c) & (A <= d)
    # If no entries match, np.mean on empty slice returns NaN, which is acceptable unless specified otherwise.
    return A[mask].mean()

def problem6(A, k):
    """
    Given an (n x n) matrix A and integer k, return an (n x k) matrix containing
    the right-eigenvectors of A corresponding to the k eigenvalues with largest magnitude.
    """
    vals, vecs = np.linalg.eig(A)
    idx = np.argsort(-np.abs(vals))[:k]
    return vecs[:, idx]

def problem7(A, x):
    """
    Given square matrix A and column vector x, compute A^{-1} x using np.linalg.solve.
    Do not form the inverse explicitly.
    """
    return np.linalg.solve(A, x)

def problem8(x, k):
    """
    Given an n-vector x and non-negative integer k, return an (n x k) matrix
    consisting of k copies of x.
    """
    return np.repeat(np.atleast_2d(x).T, k, axis=1)

def problem9(A):
    """
    Given a matrix A with n rows, return a new matrix with rows randomly permuted.
    Do not modify A.
    """
    perm = np.random.permutation(A.shape[0])
    return A[perm, :]

def problem10(A):
    """
    Given an (m x n) matrix A, return the m-vector of row means.
    """
    return A.mean(axis=1)

def problem11(n, k):
    """
    Given positive integers n and k, generate a 1-D array of n random integers from [0, k],
    then set all even numbers to -1 and return the result.
    """
    arr = np.random.randint(0, k + 1, size=n)
    arr[arr % 2 == 0] = -1
    return arr

def problem12(A, b):
    """
    Given a matrix A with n rows and an n-vector b, add b to every column of A and return the result.
    """
    b = np.asarray(b).reshape(-1, 1)
    return A + b

def problem13(A):
    """
    Given a 3-D array A of shape (n, m, m) representing n images of size m x m,
    return a 2-D array B of shape (m*m, n) such that each column i contains image i
    flattened in row-major order.
    """
    n = A.shape[0]
    return A.reshape(n, -1).T
