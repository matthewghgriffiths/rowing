from functools import partial

import numpy as np
from scipy import linalg

import jax
from jax import numpy as jnp, scipy as jsp
from jax.experimental import host_callback


from rowing.model import banded
from rowing.model.banded import BandedMatrix, bands

jax.config.update("jax_enable_x64", True)


def check(A, B, **kwargs):
    if isinstance(A, banded.BandedMatrix):
        A = A.dense()
    if isinstance(B, banded.BandedMatrix):
        B = B.dense()

    assert np.allclose(A, B, **kwargs)
    return True


def test_from_dense():
    np.random.seed(1)
    groups = [
        ((6, 8), (-2, 3)),
        ((6, 8), (-5, 6)),
        ((8, 6), (-3, 2)),
        ((8, 6), (-7, 5)),
    ]
    for (m, n), (l, u) in groups:
        A = np.random.randn(m, n)  # .round(2)
        Ab = BandedMatrix.from_dense(A, l, u)
        check(bands(A, l, u), Ab)


def test_matmul():
    np.random.seed(1)

    n = 10

    A = banded.bands(np.random.randn(n, n), -3, 1)
    B = banded.bands(np.random.randn(n, n), -2, 3)

    Ab = BandedMatrix.from_dense(A, -3, 1)
    Bb = BandedMatrix.from_dense(B, -2, 3)
    b = np.random.randn(n)

    def func(A, B, Ab, Bb):
        Ab = Ab.resolve()
        Bb = Bb.resolve()
        return banded.bands(A, Ab.l, Ab.u) @ banded.bands(B, Bb.l, Bb.u)

    def func1(A, B):
        return A @ B

    groups = [
        ((Ab, A), (Bb, B)),
        ((Ab, A), (Bb.T, B.T)),
        ((Ab.T, A.T), (Bb, B)),
        ((Ab.T, A.T), (Bb.T, B.T)),
        ((Ab, A), (Bb.transpose(), B.T)),
        ((Ab.transpose(), A.T), (Bb, B)),
        ((Ab.transpose(), A.T), (Bb.transpose(), B.T)),
    ]
    for i, ((_Ab, _A), (_Bb, _B)) in enumerate(groups):
        check(_Ab, _A)
        check(_Bb, _B)
        check(_Ab @ _Bb, _A @ _B)

        AB, vjp = jax.vjp(func, _A, _B, _Ab, _Bb)
        AB1, vjp1 = jax.vjp(func1, _Ab, _Bb)

        G = np.random.randn(n, n)
        Gb = BandedMatrix.from_dense(G, AB1.l, AB1.u)
        check(AB1, AB)
        all(map(check, vjp1(Gb), vjp(G)))


def _test_solve_triangular():
    np.random.seed(1)

    n = 50
    k = 3

    L = banded.bands(np.random.randn(n, n), -k, 0)
    i = np.arange(n)
    L = L.at[i, i].set(L[i, i]**2 + 1)
    Lb = banded.BandedMatrix.from_dense(L, -k, 0)
    A = L @ L.T
    Ab = Lb @ Lb.T
    Sl = Ab.get_bands(Ab.l, 0)
    Su = Ab.get_bands(0, Ab.u)

    Rb = banded.cholesky_banded(Su)

    check(Ab, A)
    check(banded.cholesky_banded(Sl), L, atol=1e-5)
    check(Rb, L.T, atol=1e-5)

    b = np.random.randn(n)

    def func(L, b, k, lower, trans):
        L = banded.bands(L, -k if lower else 0, 0 if lower else k)
        return jsp.linalg.solve_triangular(L, b, lower=lower, trans=trans)

    def func2(L, b, k, lower, trans):
        L = banded.bands(L, -k if lower else 0, 0 if lower else k)
        if trans:
            return L.T @ b
        return L @ b

    groups = [
        (Lb, L, True, 0),
        (Rb, L.T, False, 0),
        (Lb.T, L, True, 1),
        (Rb.T, L.T, False, 1),
    ]
    for _Lb, _L, lower, trans in groups:
        check(
            banded.solve_triangular_banded(_Lb, b),
            jsp.linalg.solve_triangular(_L, b, lower=lower, trans=trans)
        )
        x, vjp = jax.vjp(
            partial(func, k=k, lower=lower, trans=trans), _L, b
        )
        x1, vjp1 = jax.vjp(banded.solve_triangular_banded, _Lb, b)
        g = np.random.randn(*x.shape)

        check(x1, x)
        if not trans:
            all(map(check, vjp1(g), vjp(g)))
        else:
            gL1, gb1 = vjp1(g)
            gL, gb = vjp(g)
            check(gL1, gL.T)
            check(gb1, gb)

        x, vjp = jax.vjp(
            partial(func2, k=k, lower=lower, trans=trans), _L, b
        )
        x1, vjp1 = jax.vjp(banded.banded_triangular_matmul, _Lb, b)
        g = np.random.randn(*x.shape)

        check(x1, x)
        if not trans:
            all(map(check, vjp1(g), vjp(g)))
        else:
            gL1, gb1 = vjp1(g)
            gL, gb = vjp(g)
            check(gL1, gL.T)
            check(gb1, gb)


def _test_cholesky():
    n = 50
    k = 3
    np.random.seed(1)

    L = banded.bands(np.random.randn(n, n), -k, 0)
    i = np.arange(n)
    L = L.at[i, i].set(L[i, i]**2 + 1)
    Lb = banded.BandedMatrix.from_dense(L, -k, 0)
    A = L @ L.T
    Ab = Lb @ Lb.T
    Sl = Ab.get_bands(Ab.l, 0)
    Su = Ab.get_bands(0, Ab.u)

    def func(A, k):
        A = banded.bands(A, -k, k)
        return jnp.linalg.cholesky(A)

    L, vjp = jax.vjp(partial(func, k=k), A)
    dL = np.random.randn(n, n)
    dA, = vjp(dL)

    Lb, vjp1 = jax.vjp(banded.cholesky_banded, Sl)
    dSb, = vjp1(Lb.set(dL))
    check(banded.bands(dA, -k, 0), dSb)

    Rb, vjp1 = jax.vjp(banded.cholesky_banded, Su)
    dSu, = vjp1(Rb.set(dL.T))
    check(banded.bands(dA, 0, k), dSu)
