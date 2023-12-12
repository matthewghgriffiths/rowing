


# import pytest

import numpy as np
from scipy import stats

from rowing.model.gp import linalg


def make_tridiagonal(nblock, blocksize):
    n = blocksize * nblock 
    W = stats.wishart(blocksize, np.eye(blocksize))

    A = np.zeros((n,)*2)
    W = stats.wishart(blocksize ** 2, np.eye(blocksize * 2))
    for i in range(nblock - 1):
        i0 = blocksize * i
        i1 = i0 + blocksize * 2
        A[i0:i1, i0:i1] += W.rvs()

    return A 

def test_block_tridiagonal():
    np.random.seed(1)

    nblock = 4
    blocksize = 5
    n = blocksize * nblock 

    A = make_tridiagonal(nblock, blocksize)
    D = linalg.block_diag(A, blocksize, 0)
    D1 = linalg.block_diag(A, blocksize, 1)

    assert np.allclose(linalg.block_tridiagonal(D, D1), A)

    DL, DL1 = linalg.cholesky_block_tridiagonal(D, D1)
    L = linalg.set_block_diag(linalg.block_diag(DL), DL1, k=-1)
    assert np.allclose(L, np.linalg.cholesky(A), rtol=1e-10, atol=1e-6)

    y = np.random.randn(n)
    x = linalg.solve_block_triangular_bidiagonal(DL, DL1, y, lower=True)
    assert np.allclose(np.linalg.solve(L, y), x, rtol=1e-10, atol=1e-6)
    x = linalg.solve_block_triangular_bidiagonal(DL, DL1, y, lower=True, trans=1)
    assert np.allclose(np.linalg.solve(L.T, y), x, rtol=1e-10, atol=1e-6)

    DU = DL.swapaxes(1, 2)
    DU1 = linalg.block_transpose(DL1)
    x = linalg.solve_block_triangular_bidiagonal(DU, DU1, y, lower=False)
    assert np.allclose(np.linalg.solve(L.T, y), x, rtol=1e-10, atol=1e-6)
    x = linalg.solve_block_triangular_bidiagonal(DU, DU1, y, lower=False, trans=1)
    assert np.allclose(np.linalg.solve(L, y), x, rtol=1e-10, atol=1e-6)


def make_pentadiagonal(nblock, blocksize):
    D = np.zeros((nblock, blocksize, blocksize))
    D1 = np.zeros((nblock - 1, blocksize, blocksize))
    D2 = np.zeros((nblock - 2, blocksize, blocksize))
    W = stats.wishart(blocksize **2 * 3, np.eye(blocksize * 3))
    for i in range(nblock - 2):
        A = W.rvs()
        D[i] += A[:blocksize, :blocksize]
        D[i + 1] += A[blocksize:2 * blocksize, blocksize:2 * blocksize]
        D[i + 2] += A[2*blocksize:, 2*blocksize:]
        D1[i] = A[blocksize:2*blocksize, :blocksize].T
        D1[i + 1] = A[2 * blocksize:, blocksize:2 * blocksize].T
        D2[i] = A[blocksize:2*blocksize, 2 * blocksize:].T

    # D[:, np.arange(blocksize), np.arange(blocksize)] += 0#nblock
    A = linalg.block_diag(D)
    A = linalg.set_block_diag(A, D1, 1)
    A = linalg.set_block_diag(A, D2, 2)
    A = linalg.set_block_diag(A, D1, -1, 1)
    A = linalg.set_block_diag(A, D2, -2, 1)

    return A, D, D1, D2


def test_block_pentadiagonal():
    np.random.seed(2)

    nblock = 6
    blocksize = 4
    A, D, D1, D2 = make_pentadiagonal(nblock, blocksize)

    L = np.linalg.cholesky(A)
    DL, DL1, DL2 = L_Ds = linalg.cholesky_block_pentadiagaonal(D, D1, D2)
    assert np.allclose(DL, linalg.block_diag(L, blocksize))
    assert np.allclose(DL1, linalg.block_diag(L, blocksize, -1))
    assert np.allclose(DL2, linalg.block_diag(L, blocksize, -2))

    y = np.random.randn(nblock * blocksize)
    x = linalg.solve_block_triangular_tridiagonal(DL, DL1, DL2, y, lower=True)
    assert np.allclose(np.linalg.solve(L, y), x, rtol=1e-10, atol=1e-6)
    x = linalg.solve_block_triangular_tridiagonal(DL, DL1, DL2, y, lower=True, trans=1)
    assert np.allclose(np.linalg.solve(L.T, y), x, rtol=1e-10, atol=1e-6)

    U_Ds = tuple(map(linalg.block_transpose, L_Ds))
    x = linalg.solve_block_triangular_tridiagonal(*U_Ds, y, lower=False)
    assert np.allclose(np.linalg.solve(L.T, y), x, rtol=1e-10, atol=1e-6)
    x = linalg.solve_block_triangular_tridiagonal(*U_Ds, y, lower=False, trans=1)
    assert np.allclose(np.linalg.solve(L, y), x, rtol=1e-10, atol=1e-6)


