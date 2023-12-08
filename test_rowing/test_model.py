


# import pytest

import numpy as np
from scipy import stats

from rowing.model.gp import linalg


def make_tridiagonal(nblock, blocksize):
    n = blocksize * nblock 
    W = stats.wishart(blocksize, np.eye(blocksize))

    A = np.zeros((n,)*2)
    W = stats.wishart(blocksize * 2, np.eye(blocksize * 2))
    for i in range(blocksize - 2):
        i0 = blocksize * i
        i1 = i0 + blocksize * 2
        A[i0:i1, i0:i1] += W.rvs()

    return A 

def test_linalg():
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
    x = linalg.solve_block_tridiagonal(DL, DL1, y, lower=True)
    assert np.allclose(np.linalg.solve(L, y), x, rtol=1e-10, atol=1e-6)
    x = linalg.solve_block_tridiagonal(DL, DL1, y, lower=True, trans=1)
    assert np.allclose(np.linalg.solve(L.T, y), x, rtol=1e-10, atol=1e-6)

    DU = DL.swapaxes(1, 2)
    DU1 = DL1.swapaxes(1, 2)
    x = linalg.solve_block_tridiagonal(DU, DU1, y, lower=False)
    assert np.allclose(np.linalg.solve(L.T, y), x, rtol=1e-10, atol=1e-6)
    x = linalg.solve_block_tridiagonal(DU, DU1, y, lower=False, trans=1)
    assert np.allclose(np.linalg.solve(L, y), x, rtol=1e-10, atol=1e-6)
