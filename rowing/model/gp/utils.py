
from math import prod

import numpy 

import jax 
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree
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


def solve_triangular(A, b, **kwargs):
    return jax.vmap(
        lambda b: jsp.linalg.solve_triangular(A, b, **kwargs), 0, 
    )(b.reshape(b.shape[0], -1).T).T.reshape(b.shape)


def transform(func):
    return hk.without_apply_rng(hk.transform(func))

def init_apply(func, *args, **kwargs):
    func_t = transform(func)
    params = func_t.init(None, *args, **kwargs)
    return func_t.apply(params, *args, **kwargs), params

def apply(func, *args, **kwargs):
    return init_apply(func, *args, **kwargs)[0]


def _2d(x):
    x = jnp.asarray(x)
    n, *dims = x.shape
    return x.reshape(n, *(dims or (1,)))


def to_2d(*args):
    if len(args) == 1:
        return _2d(*args)
    else:
        return tuple(map(_2d, args))


class MatrixProduct:
    def __init__(self, subscripts, *operands):
        self.operands = operands
        inputs, *out = subscripts.split("->", 1)
        self.indices = inputs.split(",")
        self.subscripts = out[0] if out else "".join(self.dims)
        
    @property
    def dims(self):
        return {
            i: n for A, ind in zip(self.operands, self.indices)
            for i, n in zip(ind, A.shape)
        }

    @property
    def shape(self):
        return tuple(self.dims[i] for i in self.subscripts)

    @property 
    def size(self):
        return prod(self.shape)

    def einsum(self, *args, **kwargs):
        subscript, operands = self.norm_subscripts(*args)
        return jnp.einsum(subscript, *operands, **kwargs)

    def expand_subscript(self, mat_subscript):
        replace = dict(zip(self.subscripts, mat_subscript))
        return ["".join(replace[i] for i in ind) for ind in self.indices]

    def norm_subscripts(self, subscripts, *args):        
        inputs, *out = subscripts.split("->", 1)
        operands = (self,) + args
        operand_subs = inputs.split(",")
        expanded_subs, expanded_ops = map(
            lambda l: sum(l, []),
            zip(*(
                (M.expand_subscript(sub), list(M.operands)) 
                if isinstance(M, MatrixProduct) else ([sub], [M])
                for M, sub in zip(operands, operand_subs)
            ))
        )
        norm_subscripts = "->".join([",".join(expanded_subs)] + out)
        return norm_subscripts, expanded_ops

    def __repr__(self):
        cls = type(self)
        sub = ",".join(self.indices) + "->" + self.subscripts
        return f"{cls.__name__}({sub!r}, shape={self.shape})"
        
    def sum(self, axis=None):
        if isinstance(axis, int):
            axis=(axis,)

        subscripts = self.subscripts

        if axis is None:
            out = ""
        elif isinstance(axis, tuple):
            exclude = set(subscripts[i] for i in axis)
            out = "".join(i for i in subscripts if i not in exclude)
        else:
            raise ValueError(f"axis={axis} not a valid argument")

        return self.einsum(subscripts + "->" + out)

    @property 
    def values(self):
        return self.einsum(self.subscripts + "->" + self.subscripts)

    def __getitem__(self, index):
        index = jnp.index_exp[index]
        sub_index = {}
        for i, j in zip(index, self.subscripts):
            if i is Ellipsis:
                for i, j in zip(index[::-1], self.subscripts[::-1]):
                    if i is Ellipsis:
                        break
                    sub_index[j] = i
                break
            sub_index[j] = i

        new_operands = (
            op[tuple(sub_index.get(j, slice(None)) for j in ind)]
            for op, ind in zip(self.operands, self.indices)
        )
        new_indices = ",".join(
            "".join(j for j in ind if not jnp.isscalar(sub_index.get(j)))
            for ind in self.indices
        )
        out = "".join(j for j in self.subscripts if not jnp.isscalar(sub_index.get(j)))
        return MatrixProduct(new_indices + "->" + out, *new_operands)

    def diagonal(self, subscripts=None, axis1=None, axis2=None):
        subscripts = subscripts or self.subscripts
        axis1 = axis1 or subscripts[-2]
        axis2 = axis2 or subscripts[-1]

        def diagonals():
            for M, sub in zip(self.operands, self.expand_subscript(subscripts)):
                if axis2 in sub:
                    if axis1 in sub:
                        M = M.diagonal(axis1=sub.index(axis1), axis2=sub.index(axis2))
                        sub = sub.replace(axis2, "")
                    sub = sub.replace(axis2, axis1)
                yield M, sub

        new_operands, indices = zip(*diagonals())
        out = self.subscripts.replace(axis2, "")
        return MatrixProduct(",".join(indices) + "->" + out, *new_operands)


class OptTransform:
    def __init__(self, func, unravel, *args, _sign=-1, **kwargs):
        self.func = func
        self.unravel = unravel
        self.args = args
        self.kwargs = kwargs
        self.sign = _sign
        self._grad = jax.grad(func)

    @classmethod
    def from_transform(cls, transform, *args, _sign=-1, _rng=None, **kwargs):
        params = transform.init(_rng, *args, **kwargs)
        x, unravel = ravel_pytree(params)
        print(kwargs, "from_transform")
        return cls(transform.apply, unravel, *args, **kwargs, _sign=_sign)

    def __call__(self, x):
        params = self.unravel(x)
        return self.sign * self.func(params, *self.args, **self.kwargs)

    def func_jac(self, x):
        params = self.unravel(x)
        l, vjp = jax.vjp(lambda p: self.func(p, *self.args, **self.kwargs), params)
        grad = vjp(1.)
        G = self.sign * self.ravel(grad)
        return self.sign * l, G

    def jac(self, x):
        params = self.unravel(x)
        grad = self._grad(params, *self.args, **self.kwargs)
        return self.sign * self.ravel(grad)

    @staticmethod
    def ravel(params):
        return numpy.array(ravel_pytree(params)[0], float)

    def minimize(self, params, jac=True, method='L-BFGS-B', **kwargs):
        from scipy import optimize

        x0 = self.ravel(params)
        if jac:
            res = optimize.minimize(self.func_jac, x0, method=method, jac=True, **kwargs)
        else:
            res = optimize.minimize(self, x0, method=method, jac=self.jac, **kwargs)

        res['params'] = self.unravel(res.x)
        res['gradient'] = self.unravel(res.jac)
        return res


