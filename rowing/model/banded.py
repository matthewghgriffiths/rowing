
from typing import Optional, NamedTuple, Union
from typing_extensions import Self
from functools import partial, singledispatchmethod

import numpy as np
from scipy import linalg

import jax
from jax import numpy as jnp, scipy as jsp
from jax.experimental import host_callback

# flake8: noqa: E741

Array = Union[jax.Array, np.ndarray]


class _Banded(NamedTuple):
    bands: jax.Array
    l: int  # lower bandwidth
    u: int  # upper bandwidth
    trans: int = 0
    m: Optional[int] = None


class Banded(_Banded):
    @property
    def nblocks(self):
        return self.bands.shape[0]

    @property
    def shape(self):
        n = self.bands.shape[1]
        return (self.m, n)

    @classmethod
    def from_dense(cls, A: jax.Array, l: int, u: int) -> Self:
        return cls(*full_to_banded(A, l, u, zero=True))

    @classmethod
    def from_banded(cls, ab) -> Self:
        if isinstance(ab, cls):
            return ab
        return cls(*ab)

    @property
    def lower_triangular(self):
        return self.u == 0

    @property
    def upper_triangular(self):
        return self.l == 0

    @property
    def triangular(self):
        return self.lower_triangular or self.upper_triangular

    @property
    def T(self) -> Self:
        return self._replace(trans=int(not self.trans))

    def ones_like(self, **kwargs) -> Self:
        return self._replace(
            bands=jnp.ones_like(self.bands, **kwargs)
        ).zero_bands()

    def zeros_like(self, **kwargs) -> Self:
        return self._replace(
            bands=jnp.zeros_like(self.bands, **kwargs)
        ).zero_bands()

    def full_like(self, fill_value, **kwargs) -> Self:
        return self._replace(
            bands=jnp.full_like(self.bands, fill_value, **kwargs)
        ).zero_bands()

    def dense(self, zero=False) -> jax.Array:
        if zero:
            self = self.zero_bands()
        return _banded_to_full(*self)

    def set(self, dense: jax.Array) -> Self:
        if self.trans:
            dense = dense.T
        return self._replace(
            bands=_full_to_banded(dense, self.l, self.u, zero=True))

    def get_bands(self, l: int, u: int) -> Self:
        banded = self

        if self.trans:
            banded = banded.transpose(zero=False)

        l0 = max(0, l - banded.l)
        u0 = max(0, banded.u - u)
        bands0 = jnp.pad(
            self.bands,
            ((max(0, - u0), max(0, - l0)), (0, 0)),
        )[u0:u0 + u - l + 1]
        return banded._replace(bands=bands0, l=l, u=u)

    def transpose(self, zero: bool = False) -> Self:
        banded = self

        if zero:
            banded = banded.zero_bands()
        if banded.trans:
            return banded._replace(trans=0)
        else:
            return banded._replace(
                bands=_calc_transpose(*banded),
                l=-banded.u, u=-banded.l, trans=0
            )

    def resolve(self, zero: bool = False) -> Self:
        banded = self

        if zero:
            banded = banded.zero_bands()
        if banded.trans:
            return banded._replace(
                bands=_calc_transpose(*banded),
                l=-banded.u, u=-banded.l, trans=0
            )
        else:
            return banded

    def banded_to_full(self, zero: bool = False) -> jax.Array:
        if zero:
            self = self.zero_bands()

        return _banded_to_full(*self)

    def zero_bands(self) -> Self:
        bands, l, u, *_ = self
        nbands, n = bands.shape
        return self._replace(
            bands=jnp.where(mask_bands(l, u, n, n), bands, 0))

    def matmul_banded(self, other: _Banded, lc=None, uc=None) -> Self:
        cls = type(self)
        return cls(
            *_banded_matmul_banded(
                *self.transpose(),
                *cls.from_banded(other).resolve(),
                lc, uc
            )
        )

    def matmul(self, other: Array, check_finite=False) -> jax.Array:
        return banded_matmul(self, other, check_finite=check_finite)

    @singledispatchmethod
    def __matmul__(self, other):
        raise ValueError(f"Invalid type: {type(other)}")

    @__matmul__.register
    @__matmul__.register(jax.Array)
    def _(self, other: np.ndarray) -> jax.Array:
        return self.matmul(other)

    @__matmul__.register
    def _(self, other: _Banded) -> Self:
        return self.matmul_banded(other)

    def sum(self, axis=None, **kwargs) -> jax.Array:
        if axis is None:
            return self.bands.sum(**kwargs)
        elif axis == 0:
            return self.bands.sum(axis=0, **kwargs)
        elif axis == 1:
            return self.transpose().bands.sum(axis=0, **kwargs)
        else:
            raise ValueError(
                f"axis {axis} is out of bounds for array of dimension 2")

    def solve(self, other: Array, **kwargs) -> jax.Array:
        if self.triangular:
            return solve_triangular_banded(self, other, **kwargs)
        return solve_banded(self, other, **kwargs)

    def diagonal(self):
        return self.bands[self.u]


@jax.tree_util.register_pytree_node_class
class BandedMatrix(Banded):
    """
    Represents a banded matrix.

    Args:
        bands: The array containing the bands of the matrix.
        l: The lower bandwidth of the matrix.
        u: The upper bandwidth of the matrix. If not provided, 
            it is calculated based on the shape of the bands array.
        trans: Indicates whether the matrix is transposed.
        m: The number of rows in the matrix. If not provided, 
            it is the number of columns in the bands array.

    Attributes:
        nblocks: The number of blocks in the matrix.
        shape: The shape of the matrix.
        T: The transpose of the matrix.

    Methods:
        tree_flatten(): 
            For JAX pytree flattening.
        tree_unflatten(cls, aux_data, children): 
            For JAX pytree unflattening.
        ones_like(**kwargs) -> 'BandedMatrix': 
            Returns a BandedMatrix object with the same shape as the current object, 
            but with all elements set to 1.
        zeros_like(**kwargs) -> 'BandedMatrix': 
            Returns a BandedMatrix object with the same shape as the current object, 
            but with all elements set to 0.
        full_like(fill_value, **kwargs) -> 'BandedMatrix': 
            Returns a BandedMatrix object with the same shape as the current object, 
            but with all elements set to the specified fill value.
        from_dense(A: jax.Array, l: int, u: int) -> 'BandedMatrix': 
            Creates a BandedMatrix object from a dense matrix.
        dense(zero=False) -> jax.Array: 
            Converts the BandedMatrix object to a dense matrix.
        set(dense): 
            Sets the values of the BandedMatrix object to the values of the specified 
            dense matrix.
        get_bands(l: int, u: int) -> 'BandedMatrix': 
            Returns a new BandedMatrix object with the specified lower 
            and upper bandwidths.
        transpose(zero=False) -> 'BandedMatrix': 
            Returns the transpose of the BandedMatrix object where the bands matrix 
            has been recalculated
        resolve(zero=False) -> 'BandedMatrix': 
            Returns the resolved BandedMatrix object, which is the transpose if the 
            matrix is transposed, and vice versa.
        banded_to_full(zero=True) -> jax.Array: 
            Converts the BandedMatrix object to a full matrix.
        zero_bands() -> "BandedMatrix": 
            Returns a new BandedMatrix object with all elements in bands
            outside the bandwidth set to 0.
        matmul_banded(other: "BandedMatrix", lc=None, uc=None) -> "BandedMatrix": 
            Performs matrix multiplication with another BandedMatrix object.
        matmul(other: Union[jax.Array, np.ndarray], check_finite=False) -> jax.Array:
            Performs matrix multiplication with another matrix.
        sum(axis=None): 
            Computes the sum of the elements along the specified axis.
    """
    def __new__(cls, bands: Array, l: int, u: Optional[int] = None, trans: int = 0, m: Optional[int] = None):
        if u is None:
            u = l + bands.shape[0] - 1
        if m is None:
            m = bands.shape[1]

        return super().__new__(cls, bands, int(l), int(u), int(trans), int(m))

    def tree_flatten(self):
        return (self[:1], self[1:])

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)


def as_banded(ab):
    if isinstance(ab, BandedMatrix):
        return ab
    return BandedMatrix(*ab)


def mask_bands(l, u, m, n):
    k, j = jnp.ix_(jnp.arange(u, l - 1, -1), jnp.arange(n))
    i = j - k
    return (i >= 0) & (i < m)


def update_bands(bands, update, l, u, m=None):
    n = bands.shape[1]
    return jnp.where(mask_bands(l, u, m or n, n), update, bands)


def update_bands_dense(bands, dense, l, u):
    m, n = dense.shape
    return update_bands(bands, _full_to_banded(dense, l, u), l, u, m)


@partial(jax.jit, static_argnames=('l', 'u'))
def bands(A: jax.Array, l: int, u: int) -> jax.Array:
    i, j = jnp.indices(A.shape, sparse=True)
    k = j - i
    return jnp.where((k <= u) & (k >= l), A, 0)


@partial(jax.jit, static_argnames=('l', 'u', 'trans', 'm'))
def _banded_to_full(ab, l, u, trans=0, m=None) -> jax.Array:
    nbands, n = ab.shape
    m = m or n
    assert -m < l <= u < n
    pad = max(0, m - nbands + 1)
    m1 = max(m, nbands - 1)
    padded = jnp.pad(ab, ((0, pad), (0, 0)))
    full = padded.T.flatten()[u:n*m1 + u].reshape(n, m1)[:n, :m]

    if trans:
        return full
    else:
        return full.T


def full_to_banded(A: jax.Array, l: int, u: int, zero: bool = True) -> BandedMatrix:
    return BandedMatrix(_full_to_banded(A, l, u, zero), int(l), int(u), 0, A.shape[0])


@partial(jax.jit, static_argnames=('l', 'u', 'zero'))
def _full_to_banded(A: jax.Array, l: int, u: int, zero: bool = True) -> jax.Array:
    m, n = A.shape
    assert -m < l <= u < n
    nbands = u - l + 1
    pad = max(0, nbands - m - 1)

    if zero:
        A = bands(A, l, u)

    A = jnp.pad(A, ((0, pad), (0, 0)))
    ab = jnp.pad(A.T.flatten(), ((u, n - u),)).reshape(n, m + 1 + pad).T
    return ab[:nbands]


@partial(jax.jit, static_argnames=('l', 'u', 'trans', 'm'))
def _calc_transpose(bands: jax.Array, l: int, u: int, trans: int = 0, m=None) -> jax.Array:
    nbands, n = bands.shape
    return jnp.pad(
        bands, ((0, 0), (u < 0, u >= 0)),
    ).flatten()[max(u, 0):max(u, 0) - nbands].reshape(bands.shape)[::-1]


@partial(jax.jit, static_argnames=("la", "ua", "ta", "ma", "lb", "ub", "tb", 'mb', 'lc', 'uc'))
def _banded_matmul_banded(abT, la, ua, ta, ma, bb, lb, ub, tb, mb, lc, uc):
    la, ua = -ua, -la
    abands, na = abT.shape
    bbands, nb = bb.shape
    assert na == mb
    assert abands == ua - la + 1
    assert bbands == ub - lb + 1

    lc0 = max(la + lb, -na + 1)
    uc0 = min(ua + ub, nb - 1)

    if lc is None:
        lc = lc0
    if uc is None:
        uc = uc0

    udiff = uc0 - uc
    ldiff = lc - lc0

    cbands = uc - lc + 1
    pad = ((bbands - 1 - ldiff, bbands - 1 - udiff,), (uc, - lc))
    abT0 = jnp.pad(abT, pad)
    cb = jax.vmap(
        lambda i: jax.vmap(jnp.dot, 1)(
            bb,
            jax.lax.dynamic_slice(
                abT0, (-bb.shape[0] - i, i), bb.shape
            )
        )
    )(jnp.arange(cbands))

    return cb, lc, uc, 0, nb


def _broadcast_blas(func, shapes, ab, b, *args, **kwargs):
    if ab.ndim == 2 and b.ndim == 1:
        x = np.array(b, order='F')
        return func(
            *shapes, ab, x, *args, **kwargs,
        )
    elif ab.ndim == 2 and b.ndim == 2:
        n = b.shape[1]
        x = np.array(b, order='F')
        for i in range(n):
            x[:, i] = func(
                *shapes, ab, x[:, i], *args, **kwargs,
            )
        return x
    # elif ab.ndim == 3 and b.ndim == 1:
    #     n = ab.shape[1]
    #     x = np.empty_like(b, shape=(n, *b.shape), order='C')
    #     x[:] = b[None]
    #     for i in range(n):
    #         x[i] = func(
    #             *shapes, ab[i], x[i], *args, **kwargs,
    #         )
    #     return x
    else:
        raise ValueError(f"Invalid shapes: {ab.shape}, {b.shape}")


def _blas_banded_matmul(args, check_finite=True):
    ab, l, u, trans, m, b, alpha = args
    n = ab.shape[1]

    if check_finite:
        ab = np.asarray_chkfinite(ab, order='F')
        b = np.asarray_chkfinite(b, order='F')
    else:
        ab = np.asarray(ab, order='F')
        b = np.asarray(b, order='F')

    gbmv, = linalg.get_blas_funcs(['gbmv'], (ab, b))
    shapes = (m, n, -l, u, alpha)
    return _broadcast_blas(gbmv, shapes, ab, b, trans=trans)


def banded_matmul(ab, b, check_finite=False):
    return _banded_matmul(as_banded(ab), b, check_finite=check_finite)


@jax.custom_vjp
def _banded_matmul(ab: BandedMatrix, b, check_finite=False):
    return host_callback.call(
        partial(_blas_banded_matmul, check_finite=check_finite),
        (*ab, b, 1),
        result_shape=jax.ShapeDtypeStruct(b.shape, b.dtype)
    )


def fwd_banded_matmul(ab, b, check_finite=False,):
    c = banded_matmul(ab, b, check_finite=check_finite)
    return c, (ab, b, check_finite)


def bwd_banded_matmul(res, g):
    (ab, b, check_finite) = res
    db = banded_matmul(ab, g, check_finite=check_finite)
    g, b = jnp.atleast_2d(g.T, b.T)
    dab = ab.set(g.T @ b)
    return dab, db, None


_banded_matmul.defvjp(fwd_banded_matmul, bwd_banded_matmul)


def _scipy_solve_banded(l_and_u, arg, check_finite=False):
    return linalg.solve_banded(l_and_u, *arg, check_finite=check_finite)


def _solve_banded(ab: BandedMatrix, b: Array, check_finite=False):
    return host_callback.call(
        partial(_scipy_solve_banded, (-ab.l, ab.u), check_finite=check_finite),
        (ab.bands, b),
        result_shape=jax.ShapeDtypeStruct(b.shape, b.dtype)
    )


def solve_banded(ab, b, check_finite=False):
    return _solve_banded(as_banded(ab), b, check_finite=check_finite)


def _blas_solve_triangular_banded(args, check_finite=True):
    (ab, u, l, trans, m), b = args
    lower = bool(u)

    if check_finite:
        ab = np.asarray_chkfinite(ab, order='F')
        b = np.asarray_chkfinite(b, order='F')
    else:
        ab = np.asarray(ab, order='F')
        b = np.asarray(b, order='F')

    tbsv, = linalg.get_blas_funcs(['tbsv'], (ab, b))
    shapes = (ab.shape[-2] - 1,)
    return _broadcast_blas(
        tbsv, shapes, ab, b, lower=lower, trans=trans, overwrite_x=True)


# @partial(jax.jit, static_argnames="check_finite")
def solve_triangular_banded(ab: _Banded, b, check_finite=False) -> jax.Array:
    return _solve_triangular_banded(as_banded(ab), b, check_finite=check_finite)


@jax.custom_vjp
def _solve_triangular_banded(ab: BandedMatrix, b, check_finite=False):
    return host_callback.call(
        partial(
            _blas_solve_triangular_banded, check_finite=check_finite),
        (ab, b),
        result_shape=jax.ShapeDtypeStruct(b.shape, b.dtype)
    )


def fwd_solve_triangular_banded(ab: BandedMatrix, b, check_finite=False,):
    x = solve_triangular_banded(ab, b, check_finite)
    return x, (x, ab, b, check_finite)


def bwd_solve_triangular_banded(
    res, g
):
    (x, ab, b, check_finite) = res
    db = solve_triangular_banded(ab.T, g)
    x, db = jnp.atleast_2d(x.T, db.T)
    dab = ab.set(-db.T @ x)
    return dab, db, None


_solve_triangular_banded.defvjp(
    fwd_solve_triangular_banded,
    bwd_solve_triangular_banded
)


def _blas_banded_triangular_matmul(args, check_finite=True,):
    ab, l, u, trans, m, b = args
    lower = bool(l)
    if check_finite:
        ab = np.asarray_chkfinite(ab, order='F')
        b = np.asarray_chkfinite(b, order='F')
    else:
        ab = np.asarray(ab, order='F')
        b = np.asarray(b, order='F')

    tbsv, = linalg.get_blas_funcs(['tbmv'], (ab, b))
    shapes = (ab.shape[-2] - 1,)
    return _broadcast_blas(
        tbsv, shapes, ab, b, lower=lower, trans=trans, overwrite_x=True)


def banded_triangular_matmul(ab: BandedMatrix, b, check_finite=False) -> jax.Array:
    return _banded_triangular_matmul(
        BandedMatrix(*ab), b,
        check_finite=check_finite
    )


@jax.custom_vjp
def _banded_triangular_matmul(ab: BandedMatrix, b, check_finite=False,):
    return host_callback.call(
        partial(
            _blas_banded_triangular_matmul, check_finite=check_finite),
        (*ab, b),
        result_shape=jax.ShapeDtypeStruct(b.shape, b.dtype)
    )


def fwd_banded_triangular_matmul(ab, b, check_finite=False,):
    c = banded_triangular_matmul(ab, b, check_finite=check_finite)
    return c, (ab, b, check_finite)


def bwd_banded_triangular_matmul(
    res, g
):
    (ab, b, check_finite) = res
    db = banded_triangular_matmul(ab.T, g, check_finite=check_finite)
    g, b = jnp.atleast_2d(g.T, b.T)
    dab = ab.set(g.T @ b)
    return dab, db, None,


_banded_triangular_matmul.defvjp(
    fwd_banded_triangular_matmul, bwd_banded_triangular_matmul)


def _cholesky_banded(
    bands, lower=True,
    check_finite=True,
    overwrite_ab=False
):
    try:
        return linalg.cholesky_banded(
            bands,
            overwrite_ab=overwrite_ab,
            lower=lower,
            check_finite=check_finite
        )
    except linalg.LinAlgError:
        return np.nan * bands


@jax.custom_vjp
def cholesky_banded(ab: BandedMatrix):
    ab = as_banded(ab)
    if ab.u and ab.l:
        raise ValueError(
            "Only lower or upper triangular matrices are supported")

    chol = host_callback.call(
        partial(
            _cholesky_banded,
            lower=bool(ab.l), check_finite=False, overwrite_ab=False
        ),
        ab.bands,
        result_shape=jax.ShapeDtypeStruct(ab.bands.shape, ab.bands.dtype)
    )
    return ab._replace(bands=chol, trans=0)


def fwd_cholesky_banded(Ab):
    Lb = cholesky_banded(Ab)
    return Lb, (Lb, Ab)


def bwd_cholesky_banded(
    res, dL
):
    Lb, Ab = res
    if Lb.l:
        Lb = Lb.T
    else:
        dL = dL.T

    LTdL = BandedMatrix(*Lb).matmul_banded(BandedMatrix(*dL))
    LTdL = LTdL.get_bands(LTdL.l, 0)
    w = jnp.ones_like(LTdL.bands, shape=(
        1 - LTdL.l, 1)).at[0, 0].set(0.5) * 0.5
    phiLTdL = LTdL._replace(bands=LTdL.bands * w).dense().T
    S = solve_triangular_banded(
        Lb, solve_triangular_banded(Lb, phiLTdL).T
    )
    dA = Ab.set(S + S.T)
    return dA,


cholesky_banded.defvjp(fwd_cholesky_banded, bwd_cholesky_banded)


def bwd_cholesky(L, Lbar):
    def Phi(A):
        return jnp.tril(A) / (1 + jnp.eye(len(A)))

    P = Phi(L.T @ Lbar)
    return Phi(
        jsp.linalg.solve_triangular(
            L, jsp.linalg.solve_triangular(
                L, (P + P.T), trans=1, lower=True
            ).T, trans=1, lower=True
        )
    )


def bwd_cholesky_scan(l, l1, u1, mR, mDC, mD, lower, Dbar, kbands):
    Lbandk, dLbandk = kbands
    if lower:
        R = jnp.triu(_banded_to_full(Lbandk[1:], l1, u1, m=mR))
        Rbar = jnp.triu(_banded_to_full(dLbandk[1:], l1, u1, m=mR))
    else:
        R = _banded_to_full(Lbandk[1:], l1, u1, m=mR)
        Rbar = _banded_to_full(dLbandk[1:], l1, u1, m=mR)

    Rbar -= (Dbar + Dbar.T) @ R
    Rbandk = update_bands_dense(dLbandk[1:], Rbar, l1, u1)
    dLbandk = dLbandk.at[1:].set(Rbandk)

    DC = _banded_to_full(Lbandk, l, 0, m=mDC)
    DCbar = _banded_to_full(dLbandk, l, 0, m=mDC)
    D, C = DC[:mD], DC[mD:]
    Dbar, Cbar = DCbar[:mD], DCbar[mD:]

    Cbar = jsp.linalg.solve_triangular(
        D, Cbar.T, lower=True, trans=1).T
    Dbar = bwd_cholesky(D, jnp.tril(Dbar - Cbar.T @ C))
    Abandk = _full_to_banded(jnp.vstack([Dbar, Cbar]), l, 0)
    return Dbar, Abandk.T


def _bwd_cholesky_lower_banded(Lb, dLb):
    nbands, n = Lb.shape

    l = 1 - nbands
    NB = -l
    u = - l
    s = NB - u
    l1, u1 = s, u - 1 + s
    remainder = n % NB

    Lbands = Lb[:, remainder:].reshape(nbands, -1, NB).swapaxes(1, 0)
    dLbands = dLb[:, remainder:].reshape(nbands, -1, NB).swapaxes(1, 0)

    Dbar = jnp.zeros((NB, NB))
    Dbar, Abands = jax.lax.scan(
        partial(bwd_cholesky_scan, l, l1, u1, NB, 2 * NB, NB, True),
        Dbar, (Lbands, dLbands), reverse=True,
    )
    u1 = min(u - 1 + s, remainder - 1)
    l1 = u1 - u + 1
    _, Aband0 = bwd_cholesky_scan(
        l, l1, u1, NB, remainder - l, remainder, False,
        Dbar, (Lb[:, :remainder], dLb[:, :remainder])
    )
    dAb = jnp.c_[Aband0.T, Abands.reshape(-1, nbands).T]

    return dAb


def _cholesky_jvp(L, sigma_dot):
    # Forward-mode rule from https://arxiv.org/pdf/1602.07527.pdf
    tmp1 = jax.lax.linalg.triangular_solve(
        L, sigma_dot, left_side=False, transpose_a=True,
        conjugate_a=True, lower=True)
    tmp2 = jax.lax.linalg.triangular_solve(
        L, tmp1, left_side=True, transpose_a=False, lower=True)
    tmp3 = jnp.tril(tmp2) / (1 + jnp.eye(L.shape[-1], dtype=tmp2.dtype))
    L_dot = jax.lax.batch_matmul(L, tmp3, precision=jax.lax.Precision.HIGHEST)
    return L, L_dot


def closest_cholesky(A, min_eig=0.):
    l, V = jnp.linalg.eigh(A)
    min_eig = jax.lax.select(
        min_eig > 0,
        min_eig,
        jnp.where(l > 0, l, l.max()).min(),
    )
    _, R = jnp.linalg.qr(
        (V * jnp.sqrt(jnp.clip(l, min_eig, None))).T
    )
    L = R.T * jnp.sign(R.diagonal())
    return L


@jax.custom_jvp
def safe_cholesky(A, min_eig=0.):
    L = jnp.linalg.cholesky(A)
    isposdef = jnp.isfinite(L).all()
    return jax.lax.cond(
        isposdef,
        lambda: L,
        lambda: closest_cholesky(A, min_eig=jnp.float_(min_eig)),
    )


@safe_cholesky.defjvp
def safe_cholesky_vjp(primals, tangents):
    A, _ = primals
    Adot, _ = tangents
    L = jnp.linalg.cholesky(A)
    isposdef = jnp.isfinite(L).all()
    return jax.lax.cond(
        isposdef,
        lambda: _cholesky_jvp(L, Adot),
        lambda: jax.jvp(closest_cholesky, primals, tangents),
    )


def _safe_cholesky_banded_scan(l, mL, mLR, R10, Abandk, min_eig=0.):
    LR = _banded_to_full(Abandk, l, 0, m=mLR)
    A0_ = LR[:mL]
    A1 = LR[mL:]
    A0 = A0_ + jnp.tril(A0_, -1).T
    L00 = safe_cholesky(A0 - R10 @ R10.T, min_eig=min_eig)
    R10 = jax.lax.linalg.triangular_solve(
        L00, A1, left_side=False, lower=True, transpose_a=True
    )
    Lbandk = _full_to_banded(jnp.vstack([L00, R10]), l, 0)
    return R10, Lbandk.T


def _safe_cholesky_banded(Abands, min_eig=0.):
    nbands, n = Abands.shape
    NB = nbands - 1
    l = - NB
    remainder = n % NB

    Abandks = Abands[:, :-remainder].reshape(nbands, -1, NB).swapaxes(1, 0)
    R10, Lbandk = jax.lax.scan(
        partial(_safe_cholesky_banded_scan, l, NB, 2 * NB, min_eig=min_eig),
        jnp.zeros((NB, NB)), Abandks
    )
    _, Lband1 = _safe_cholesky_banded_scan(
        l, remainder, 2 * NB, R10[:remainder], Abands[:, -remainder:], min_eig=min_eig)

    Lbands = jnp.c_[Lbandk.reshape(-1, nbands).T, Lband1.T]
    return Lbands


def safe_cholesky_banded(ab: Banded, min_eig=0.):
    if ab.upper_triangular:
        ab = ab.transpose()

    Lbands = _safe_cholesky_banded(ab.bands, min_eig=min_eig)
    return type(ab)(Lbands, ab.l, 0, trans=0)


def solve_cholesky_banded(Lb: Banded, other):
    if Lb.lower_triangular:
        return Lb.T.solve(Lb.solve(other))
    elif Lb.upper_triangular:
        return Lb.solve(Lb.T.solve(other))

    raise ValueError("Banded matrix not triangular")
