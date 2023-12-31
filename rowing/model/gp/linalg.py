
from typing import NamedTuple, Tuple
from functools import partial 

import numpy as np

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

def maximum(x, *args):
    for y in args:
        x = jnp.maximum(x, y)
    return x

def minimum(x, *args):
    for y in args:
        x = jnp.minimum(x, y)
    return x


def _tot_blocks(nblocks, k0, k1):
    return (
        nblocks * (k1 - k0 + 1) - (k0 * (k0 - 1)) // 2 - (k1 * (k1 + 1)) // 2
    )

def _block_matmul(A, B, args):
    i, j, ak0, ak1, kc = args
    
    def body(ka, val):
        CB = (
            A.get_diag_block(jnp.minimum(i, i + ka), ka) 
            @ B.get_diag_block(jnp.minimum(j, i + ka), kc - ka)
        )
        return val + CB
    
    res = jax.lax.fori_loop(
        ak0, ak1 + 1, body, 
        jnp.zeros(A.blockshape[:1] + B.blockshape[1:])
    )
    return res

@partial(jax.jit, static_argnums=list(range(5)))
def _blocks_matmul_indexes(nblocks, ak0, ak1, bk0, bk1):
    ck0, ck1 = ak0 + bk0, ak1 + bk1
    size = BlockBanded._tot_blocks(nblocks, ck0, ck1)
    cks, cblocksizes, cblockstart, _ = BlockBanded.block_shapes(nblocks, ck0, ck1)
    ck = jnp.repeat(cks, cblocksizes, total_repeat_length=size)
    kstart = jnp.repeat(cblockstart, cblocksizes, total_repeat_length=size)
    cn = jnp.arange(ck.size) - kstart
    ci, cj = jnp.maximum(cn - ck, cn), jnp.maximum(cn + ck, cn)
    k0 = maximum(ak0, - ci, ck - bk1, ck - cj)
    k1 = minimum(ak1, nblocks - ci - 1, ck - bk0, ck - cj + nblocks - 1)
    return (ci, cj, k0, k1, ck), (int(ck0), int(ck1))

@partial(jax.jit, static_argnums=range(2, 7))
def _block_banded_matmul(A, B, nblocks, ak0, ak1, bk0, bk1):
    args, (ck0, ck1) = _blocks_matmul_indexes(nblocks, ak0, ak1, bk0, bk1)
    return jax.lax.map(partial(_block_matmul, A, B), args)

def block_banded_matmul(A, B):
    nblocks = A.nblocks
    blocks = _block_banded_matmul(A, B, nblocks, A.k0, A.k1, B.k0, B.k1)
    return block_banded(blocks, nblocks, A.k0 + B.k0, A.k1 + B.k1)


@partial(jax.jit, static_argnums=range(1, 4))
def block_banded(blocks, nblocks, k0, k1):
    return BlockBanded(blocks, nblocks, k0, k1)

@jax.tree_util.register_pytree_node_class
class BlockBanded:
    blocks: jax.Array
    nblocks: int
    k0: int
    k1: int

    # @partial(jax.jit, static_argnums=range(2, 5))
    def __init__(self, blocks, nblocks: int, k0: int, k1: int):
        self.blocks = blocks
        self.nblocks = int(nblocks)
        self.k0 = int(k0)
        self.k1 = int(k1)
        # self.nblocks = jnp.array(nblocks, int).item()
        # self.k0 = jnp.array(k0, int).item()
        # self.k1 = jnp.array(k1, int).item()

    def tree_flatten(self):
        return ((self.blocks,), (self.nblocks, self.k0, self.k1,))
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)
    
    @staticmethod
    def _tot_blocks(nblocks, k0, k1):
        return (
            nblocks * (k1 - k0 + 1) 
            - (k0 * (k0 - 1)) // 2 
            - (k1 * (k1 + 1)) // 2
        )

    @staticmethod
    def block_coords(n, k):
        return jnp.maximum(n - k, n), jnp.maximum(n + k, n)
    
    @staticmethod
    def diag_coords(i, j):
        return jnp.minimum(i, j), j - i
    
    def get_diag_block(self, n, k):
        return self.blocks[self.blockstart[k] + n]
    
    def get_block(self, i, j):
        return self.get_diag_block(*self.diag_coords(i, j))
    
    @staticmethod
    @partial(jax.jit, static_argnums=range(3))
    def block_shapes(nblocks, k0, k1):
        ks = np.roll(np.arange(k0, k1 + 1), k0)
        blocksizes = nblocks - abs(ks)
        blockstart = np.r_[0, blocksizes[:-1].cumsum()]
        return ks, blocksizes, blockstart, BlockBanded._tot_blocks(nblocks, k0, k1)

    @classmethod
    def from_flat(cls, flatblocks, nblocks, k0, k1):
        return cls(jnp.array(flatblocks), nblocks, k0, k1)

    @classmethod
    def from_blocks(cls, blocks, k0):
        nblocks = len(blocks[0])
        blockshape = blocks[0][0].shape
        k1 = len(blocks) + k0 - 1
        ks, blocksizes, blockstart, blocktot = cls.block_shapes(nblocks, k0, k1)
        flat_blocks = jnp.zeros((blocktot,) + blockshape)
        for k in ks:
            for i, M in enumerate(blocks[k]):
                flat_blocks = flat_blocks.at[blockstart[k] + i].set(M)

        return cls(flat_blocks, nblocks, k0, k1)
    
    @classmethod
    def block_diag_indexes(cls, nblocks, k0, k1):
        ks, blocksizes, blockstart, size = cls.block_shapes(nblocks, k0, k1)
        k = jnp.repeat(ks, blocksizes, total_repeat_length=size)
        n = jnp.arange(size) - jnp.repeat(blockstart, blocksizes, total_repeat_length=size)
        return n, k

    @classmethod
    def from_dense(cls, A, blockshape, k0, k1):
        n0, n1 = blockshape
        nblocks = A.shape[0] // n0
        assert A.shape[1] // n1 == nblocks
        ks, blocksizes, blockstart, size = cls.block_shapes(nblocks, k0, k1)
        k = jnp.repeat(ks, blocksizes, total_repeat_length=size)
        n = jnp.arange(size) - jnp.repeat(blockstart, blocksizes, total_repeat_length=size)
        i, j = cls.block_coords(n, k)

        def Aindex(args):
            return jax.lax.dynamic_slice(A, args, blockshape)
            # return A[i*n0:i*n0 + n0, j*n1:j*n1 + n1]

        blocks = jax.lax.map(Aindex, (n0 * i, n1 * j))
        return cls(blocks, nblocks, k0, k1)
        
        # A = jnp.zeros(self.shape, self.dtype)
        # n, m = self.blockshape
        # for (i, j), M in self.items():
        #     A = A.at[i*n:i*n + n, j*m:j*m + m].set(M)
        # return A



    @property 
    def dtype(self):
        return self.blocks.dtype
    
    @property
    def blockshape(self):
        return self.blocks.shape[1:]
    
    @property 
    def tot_blocks(self):
        return len(self.blocks)
    
    @property 
    def shape(self):
        nblocks = self.nblocks
        return tuple(n * nblocks for n in self.blockshape)
    
    @property 
    def ks(self):
        return jnp.roll(jnp.arange(self.k0, self.k1 + 1), self.k0)
    
    @property 
    def blocksizes(self):
        return self.nblocks - abs(self.ks)

    @property 
    def blockstart(self):
        return jnp.r_[0, self.blocksizes[:-1].cumsum()]

    def items(self):
        for k, kstart in zip(self.ks, self.blockstart):
            for n in range(self.nblocks - abs(k)):
                yield self.block_coords(n, k), self.blocks[kstart + n]


    def diag_indexes(self):
        ks, blocksizes, blockstart, size = self.block_shapes(self.nblocks, self.k0, self.k1)
        k = jnp.repeat(ks, blocksizes, total_repeat_length=len(self.blocks))
        n = jnp.arange(len(self.blocks)) - jnp.repeat(blockstart, blocksizes, total_repeat_length=len(self.blocks))
        return n, k

    def dense(self):
        A = jnp.zeros(self.shape, self.dtype)
        n, k = self.diag_indexes()
        i, j = self.block_coords(n, k)
        n, m = self.blockshape
        
        ni, mj = n * i, m * j

        def body(k, A):
            return jax.lax.dynamic_update_slice(A, self.blocks[k], (ni[k], mj[k]))
        return jax.lax.fori_loop(0, len(self.blocks), body, A)
    
    def __matmul__(self, other):
        if isinstance(other, BlockBanded):
            return block_banded_matmul(self, other)
        raise NotImplementedError()
    
    @property
    def T(self):
        nblocks, k0, k1 = self.nblocks, self.k0, self.k1
        n, k = BlockBanded.block_diag_indexes(nblocks, -k1, -k0)
        _, _, blockstart, _ = BlockBanded.block_shapes(nblocks, k0, k1)
        return BlockBanded(
            block_transpose(self.blocks)[n + blockstart[- k]], 
            nblocks, -k1, -k0
        )


block_transpose = jax.vmap(jnp.transpose)

class SymmetricBlockBanded(BlockBanded):
    def get_diag_block(self, n, k):
        return jax.lax.cond(
            k < 0, 
            lambda n, k: self.blocks[self.blockstart[-k] + n].T, 
            lambda n, k: self.blocks[self.blockstart[k] + n]
        )



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
            *cholesky_block_tridiagonal(*self, min_eig=min_eig)
        )


class SymmetricBlockPentadiagonal(NamedTuple):
    D: jax.Array 
    D1: jax.Array 
    D2: jax.Array

    def cholesky(self, min_eig=-1.):
        return BlockTriangularTridiagonal(
            *cholesky_block_pentadiagaonal(*self, min_eig=min_eig)
        )

