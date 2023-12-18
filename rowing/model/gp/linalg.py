
from typing import NamedTuple
from functools import partial 

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import haiku as hk

vdot = jax.vmap(jnp.dot)


def get_pos_def(n, dof=None, log_diag=0., name="pos_def"):
    diag = jnp.exp(hk.get_parameter(
        f"{name}_diag", shape=(n,), dtype="f", init=lambda s, d: jnp.full(s, log_diag, d)
    ))
    P = jnp.diag(diag)
    if dof:
        W = hk.get_parameter(
            f"{name}_W", shape=(n, dof), dtype='f', init=jnp.zeros
        )
        return W.dot(W.T) + P
    else:
        return P


def solve_triangular(A: jax.Array, b: jax.Array, **kwargs):
    return jax.vmap(
        lambda b: jsp.linalg.solve_triangular(A, b, **kwargs), 0,
    )(b.reshape(b.shape[0], -1).T).T.reshape(b.shape)



partial(jax.jit, static_argnames=('min_eig',))
def closest_cholesky(A, min_eig=0.):
    l, V = jnp.linalg.eigh(A)
    min_eig = jax.lax.select(
        min_eig < 0, 
        jnp.where(l > 0, l, l.max()).min(), 
        min_eig
        # jnp.asarray(min_eig, dtype=l.dtype)
    )

    _, R = jnp.linalg.qr(
        (V * jnp.sqrt(jnp.clip(l, min_eig, None))).T
    )
    L = R.T * jnp.sign(R.diagonal())
    return L 


partial(jax.jit, static_argnames=('min_eig',))
def cholesky(A, min_eig=0):
    L = jnp.linalg.cholesky(A)
    isposdef = jnp.isfinite(L[0, 0])
    return jax.lax.cond(
        isposdef, 
        lambda A: L, 
        lambda A: closest_cholesky(A, min_eig=jnp.float_(min_eig)), 
        A
    )

block_transpose = jax.vmap(jnp.transpose)


@partial(jax.jit, static_argnames=['k', 'trans'])
def set_block_diag(A, D, k=0, trans=0):
    blocksize = jnp.shape(D)[1]
    if k >= 0:
        for i, Di in enumerate(D):
            Di = Di.T if trans else Di
            i0 = i * blocksize
            i1 = i0 + blocksize 
            j0 = i0 + k * blocksize
            j1 = i1 + k * blocksize
            A = A.at[i0:i1,j0:j1].set(Di)
    else:
        for i, Di in enumerate(D):
            Di = Di.T if trans else Di
            i0 = i * blocksize
            i1 = i0 + blocksize 
            j0 = i0 - k * blocksize
            j1 = i1 - k * blocksize
            A = A.at[j0:j1,i0:i1].set(Di)

    return A 


def block_tridiagonal(D, D1, upper=True):
    A = jsp.linalg.block_diag(*D)
    return set_block_diag(
        set_block_diag(A, D1, k=-1, trans=upper), 
        D1, k=1, trans=not upper
    )


def block_diag(A, blocksize=None, k=0):
    if blocksize:
        kblock = k * blocksize
        n = len(A)
        if k >= 0:
            return jnp.array([
                A[i0:i0 + blocksize, i0 + kblock:i0 + blocksize + kblock]
                for i0 in range(0, n - k * blocksize , blocksize)
            ])
        else:
            return jnp.array([
                A[i0 - kblock:i0 + blocksize - kblock, i0:i0 + blocksize]
                for i0 in range(0, n + k * blocksize , blocksize)
            ])
    else:
        if k == 0:
            return jsp.linalg.block_diag(*A)

        m0, m1, m2 = A.shape 
        assert m1 == m2
        blocksize = m1
        nblocks = m0 + abs(k) 
        n = nblocks * blocksize
        return set_block_diag(jnp.zeros((n, n)), A, k=k)
    
            

@partial(jax.jit, static_argnames=('lower', 'trans'))
def solve_block_triangular_bidiagonal(
    D: jax.Array, D1: jax.Array, y: jax.Array, lower=False, trans=0
):
    forward = not lower 
    if trans == 0 or trans == 'N':
        forward = lower
        def _blockbidiagscan(carry, xs):
            yi, Ui, D1i = xs 
            xi = jsp.linalg.solve_triangular(
                Ui, yi - D1i @ carry, lower=lower, trans=trans
            )
            return xi, xi 
    else:
        forward = not lower 
        def _blockbidiagscan(carry, xs):
            yi, Ui, D1i = xs 
            xi = jsp.linalg.solve_triangular(
                Ui, yi - D1i.T @ carry, lower=lower, trans=trans
            )
            return xi, xi 
        
        if trans == 2 or trans == "C":
            D1 = jnp.conjugate(D1)

    Y = jnp.reshape(y, jnp.shape(D)[:-1] + jnp.shape(y)[1:])
    if forward:
        xp = jsp.linalg.solve_triangular(
            D[0], Y[0], lower=lower, trans=trans
        )
        xs = (Y[1:], D[1:], D1)
        _, x0 = jax.lax.scan(
            _blockbidiagscan, xp, xs, reverse=False, 
        )
        X = jnp.concatenate([xp[None], x0])
    else:
        xp = jsp.linalg.solve_triangular(
            D[-1], Y[-1], lower=lower, trans=trans
        )
        xs = (Y[:-1], D[:-1], D1)
        _, x0 = jax.lax.scan(
            _blockbidiagscan, xp, xs, reverse=True
        )
        X = jnp.concatenate([x0, xp[None]])
        
    return X.reshape(y.shape)


@partial(jax.jit, static_argnames=('lower', 'trans'))
def solve_block_triangular_tridiagonal(
    D: jax.Array, D1: jax.Array, D2: jax.Array, y: jax.Array, 
    lower=False, trans=0
):
    def _scan(carry, xs):
        xp0, xp1 = carry 
        y2, DL22, DL21, DL20 = xs 
        xp2 = jsp.linalg.solve_triangular(
            DL22, y2 - DL21 @ xp1 - DL20 @ xp0, 
            lower=lower, trans=trans
        )
        return (xp1, xp2), xp2 
    
    if trans == 0 or trans == 'N':
        forward = lower
    else:
        forward = not lower
        D1 = block_transpose(D1)
        D2 = block_transpose(D2)
        if trans == 2 or trans == "C":
            D1 = jnp.conjugate(D1)
            D2 = jnp.conjugate(D2)

    Y = jnp.reshape(y, jnp.shape(D)[:-1] + jnp.shape(y)[1:])
    if forward:
        xp0 = jsp.linalg.solve_triangular(D[0], Y[0], lower=lower, trans=trans)
        xp1 = jsp.linalg.solve_triangular(
            D[1], Y[1] - D1[0] @ xp0, lower=lower, trans=trans)
        _, x2 = jax.lax.scan(
            _scan, (xp0, xp1), (Y[2:], D[2:], D1[1:], D2)
        )
        X = jnp.concatenate([xp0[None], xp1[None], x2])
    else:
        xp0 = jsp.linalg.solve_triangular(D[-1], Y[-1], lower=lower, trans=trans)
        xp1 = jsp.linalg.solve_triangular(
            D[-2], Y[-2] - D1[-1] @ xp0, lower=lower, trans=trans)
        _, x2 = jax.lax.scan(
            _scan, (xp0, xp1), (Y[:-2], D[:-2], D1[:-1], D2),
            reverse=True
        )
        X = jnp.concatenate([x2, xp1[None], xp0[None]])
        
    return X.reshape(y.shape)



partial(jax.jit, static_argnames=('min_eig',))
def cholesky_block_tridiagonal(D: jax.Array, D1: jax.Array, min_eig=-1.):
    def _blockbidiagscan(L0, xs):
        D11, D01 = xs 
        L01 = jsp.linalg.solve_triangular(L0, D01, lower=True, trans=0).T
        L11 = cholesky(D11 - L01 @ L01.T, min_eig=min_eig)
        return L11, (L11, L01)
    
    L0 = cholesky(D[0], min_eig=min_eig)
    _, (DL, DL1) = jax.lax.scan(
        _blockbidiagscan, L0, (D[1:], D1)
    )
    DL = jnp.concatenate([L0[None], DL], axis=0)
    return DL, DL1



partial(jax.jit, static_argnames=('min_eig',))
def cholesky_block_pentadiagaonal(
    D: jax.Array, D1: jax.Array, D2: jax.Array, min_eig=-1.
):
    L00 = cholesky(D[0], min_eig=min_eig)
    L01 = jsp.linalg.solve_triangular(L00, D1[0], lower=True, trans=0).T
    L11 = cholesky(D[1] - L01 @ L01.T, min_eig=min_eig)

    def _blocktridiagscan(carry, xs):
        L00, L11, L01 = carry 
        D02, D11, D20 = xs 

        L02 = jsp.linalg.solve_triangular(
            L00, D20, lower=True, trans=0).T
        L12 = jsp.linalg.solve_triangular(
            L11, D11 - L01 @ L02.T, lower=True, trans=0).T
        L22 = cholesky(
            D02 - L02 @ L02.T - L12 @ L12.T, min_eig=min_eig)
        return (L11, L22, L12), (L22, L12, L02)

    _, (dL, dL1, DL2) = jax.lax.scan(
        _blocktridiagscan, (L00, L11, L01), (D[2:], D1[1:], D2)
    )
    DL = jnp.concatenate([L00[None], L11[None], dL], axis=0)
    DL1 = jnp.concatenate([L01[None], dL1], axis=0)
    return DL, DL1, DL2



class BlockTriangularBidiagonal(NamedTuple):
    D: jax.Array 
    D1: jax.Array 

    def solve(self, y, trans=0):
        return solve_block_triangular_bidiagonal(
            *self, y, lower=True, trans=trans
        )
    

class BlockTriangularTridiagonal(NamedTuple):
    D: jax.Array 
    D1: jax.Array 
    D2: jax.Array
    
    def solve(self, y, trans=0):
        return solve_block_triangular_tridiagonal(
            *self, y, lower=True, trans=trans
        )


class SymmetricBlockTridiagonal(NamedTuple):
    D: jax.Array 
    D1: jax.Array 
    
    def cholesky(self, min_eig=-1.):
        return BlockTriangularBidiagonal(
            *cholesky_block_tridiagaonal(*self, min_eig=min_eig)
        )


class SymmetricBlockPentadiagonal(NamedTuple):
    D: jax.Array 
    D1: jax.Array 
    D2: jax.Array

    def cholesky(self, min_eig=-1.):
        return BlockTriangularTridiagonal(
            *cholesky_block_pentadiagaonal(*self, min_eig=min_eig)
        )

