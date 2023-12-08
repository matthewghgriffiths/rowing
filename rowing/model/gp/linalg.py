
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
                A[i0 + kblock:i0 + blocksize + kblock, i0:i0 + blocksize]
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
def solve_block_tridiagonal(
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

@jax.jit
def cholesky_block_tridiagonal(D: jax.Array, D1: jax.Array):
    def _blockbidiagscan(L0, xs):
        D11, D01 = xs 
        L01 = jsp.linalg.solve_triangular(L0, D01, lower=True, trans=0).T
        L11 = jnp.linalg.cholesky(D11 - L01 @ L01.T)
        return L11, (L11, L01)
    
    L0 = jnp.linalg.cholesky(D[0])
    _, (DL, DL1) = jax.lax.scan(
        _blockbidiagscan, L0, (D[1:], D1)
    )
    DL = jnp.concatenate([L0[None], DL], axis=0)
    return DL, DL1