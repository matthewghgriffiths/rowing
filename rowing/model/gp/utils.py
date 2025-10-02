
from math import prod
import yaml
from functools import partial
from typing import Dict, List, Tuple, Optional, Generator, NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax._src.flatten_util import ravel_pytree
import haiku as hk

from ...utils import map_concurrent
from .linalg import vdot, get_pos_def, solve_triangular


def norm_jax(x):
    if x.shape == ():
        return x.item()
    return x.tolist()


def is_list(x):
    if isinstance(x, list):
        return True
    return False


def dump_params(params, *args, **kwargs):
    norm_params = jax.tree_map(norm_jax, params)
    return yaml.safe_dump(norm_params, *args, **kwargs)


def load_params(obj, *args, **kwargs):
    norm_params = yaml.safe_load(obj, *args, **kwargs)
    return jax.tree_map(jnp.array, norm_params, is_leaf=is_list)


def open_params(path="params.yaml"):
    with open(path, "r") as f:
        return load_params(f)


def transform(func):
    return hk.without_apply_rng(hk.transform(func))


def init_apply(func, *args, **kwargs):
    func_t = transform(func)
    params = func_t.init(None, *args, **kwargs)
    return func_t.apply(params, *args, **kwargs), params


def apply(func, *args, **kwargs):
    return init_apply(func, *args, **kwargs)[0]


def init_full(fill_value):
    def full(shape, dtype):
        return jnp.full(shape, fill_value, dtype)

    return full


def get_full_kernel(self):
    kernels = self.get_kernels()
    return sum(kernels)


def get_jitter_kernel(self):
    K = self.get_full_kernel()
    race_var = jnp.exp(hk.get_parameter(
        "log_noise", [], init=jnp.zeros, dtype=jnp.float64))
    K_noise = jnp.eye(len(K)) * race_var
    return K + K_noise


def gp_system(self):
    K = self.get_jitter_kernel()
    y = self.y
    return GPSystem.from_gram(K, y)


def loss(self):
    return self.gp_system().loss()


class GPSystem(NamedTuple):
    a: jax.Array
    Ly: jax.Array
    L: jax.Array
    y: jax.Array

    @classmethod
    def from_cholesky(cls, L, y):
        Ly = solve_triangular(L, y, lower=True, trans=0)
        a = solve_triangular(L, Ly, lower=True, trans=1)
        return cls(a, Ly, L, y)

    @classmethod
    def from_gram(cls, K, y):
        L = jnp.linalg.cholesky(K)
        return cls.from_cholesky(L, y)

    def log_marginal(self, constant=1):
        L = self.L
        Ly = self.Ly
        log_like = constant * (
            - jnp.dot(Ly, Ly)/2
            - jnp.log(L.diagonal()).sum()
        )
        return log_like

    def loss(self):
        return self.log_marginal(-1)

    def inv_K(self):
        L, y = self.L, self.y
        invK = solve_triangular(
            L,
            solve_triangular(
                L, jnp.eye(len(y)),
                lower=True, trans=0
            ),
            lower=True, trans=1
        )
        return invK

    def leave_one_out(self):
        y = self.y
        a = self.a
        invK = self.inv_K()
        iKii = invK.diagonal()
        y_loo = y - a / iKii
        return y_loo

    def leave_groups_out(self, groups):
        y = self.y
        a = self.a
        invK = self.inv_K()
        y_lgo = np.zeros_like(a)
        for i in groups:
            ij = np.ix_(i, i)
            y_lgo[i] = y[i] - np.linalg.inv(invK[ij]) @ a[i]

        return y_lgo

    def leave_groups_out_prediction(self, groups, K):
        iK = self.inv_K()
        KiK = K @ iK
        y_pred = np.zeros_like(self.y)
        k = np.arange(len(K))
        for i in groups:
            ii = np.ix_(i, i)
            j = np.setdiff1d(k, i)
            ij = np.ix_(i, j)

            iKii = iK[ii]
            yj = self.y[j]
            y_pred[i] = (
                KiK[ij] @ yj
                - np.linalg.multi_dot([K[ii], iK[ij], yj])
                - np.linalg.multi_dot([
                    (KiK[ii] - K[ii] @ iKii),
                    np.linalg.inv(iKii), iK[ij], yj
                ])
            )

        return y_pred

    def predict(self, K):
        return K @ self.a

    def var(self, Kvar, K):
        L = self.L
        LdivK = solve_triangular(
            L, K.T, lower=True, trans=0
        )
        if jnp.ndim(Kvar) == 2:
            Kvar = Kvar.diagonal()

        y_var = Kvar - jnp.square(LdivK).sum(0)
        return y_var

    def covar(self, Kcov, K):
        L = self.L
        LdivK = solve_triangular(
            L, K.T, lower=True, trans=0
        )
        return Kcov - LdivK.T @ LdivK

    def mean_loss(self):
        return self.loss() / self.y.size

    def mahalanobis(self):
        return self.a @ self.y / self.a.size

    def mse(self):
        y_loo = self.leave_one_out()
        return jnp.square(self.y - y_loo).mean()

    def rmse(self):
        return jnp.sqrt(self.mse())


class OptModel:
    def __init__(self, model, pbar=None, callback=True):
        self.model = model
        self.system = transform(model.gp_system)
        self.loss = jax.jit(transform(self.model.loss).apply)

        self.pbar = pbar
        self._callback = callback
        self.args_history = []
        self.loss_history = []
        self.rmse_history = []

    def set_pbar(self, pbar):
        self.pbar = pbar
        return self

    def __call__(self, params, *args,  **kwargs):
        jax.debug.callback(
            self.callback, params, *args, ordered=True, **kwargs)
        return self.loss(params, *args, **kwargs)

    def callback(self, *args, **kwargs):
        if self._callback:
            self.default_callback(*args, **kwargs)

    def default_callback(self, *args, **kwargs):
        try:
            self.args_history.append(args)
            system = self.system.apply(*args, **kwargs)

            loss = float(system.mean_loss())
            self.loss_history.append(loss)
            mahalanobis = float(system.mahalanobis())
            rmse = float(system.rmse())
            self.rmse_history.append(rmse)

            if self.pbar:
                self.pbar.update(1)
                self.pbar.set_postfix(
                    loss=loss,
                    rmse=rmse,
                    Mahalanobis=mahalanobis,
                )

        except Exception as e:
            if self.pbar:
                self.pbar.set_postfix(EXCEPTION=e)


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
    def __init__(self, subscripts: str, *operands: np.ndarray):
        self.operands = operands
        inputs, *out = subscripts.split("->", 1)
        self.indices = inputs.split(",")
        self.subscripts = out[0] if out else "".join(self.dims)

    @property
    def dims(self) -> Dict[str, int]:
        return {
            i: n for A, ind in zip(self.operands, self.indices)
            for i, n in zip(ind, A.shape)
        }

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.dims[i] for i in self.subscripts)

    @property
    def size(self) -> int:
        return prod(self.shape)

    def einsum(self, subscripts: str, *args: np.ndarray, **kwargs) -> np.ndarray:
        subscript, operands = self.norm_subscripts(subscripts, *args)
        return jnp.einsum(subscript, *operands, **kwargs)

    def expand_subscript(self, mat_subscript) -> List[str]:
        replace = dict(zip(self.subscripts, mat_subscript))
        return ["".join(replace[i] for i in ind) for ind in self.indices]

    def norm_subscripts(self, subscripts: str, *args: np.ndarray):
        inputs, *out = subscripts.split("->", 1)
        operands = (self,) + args
        operand_subs = inputs.split(",")
        expanded_subs, expanded_ops = map(
            lambda l: sum(l, []),
            zip(*(
                (M.expand_subscript(sub), list(M.operands))
                if isinstance(M, type(self)) else ([sub], [M])
                for M, sub in zip(operands, operand_subs)
            ))
        )
        norm_subscripts = "->".join([",".join(expanded_subs)] + out)
        return norm_subscripts, expanded_ops

    def __repr__(self):
        cls = type(self)
        sub = ",".join(self.indices) + "->" + self.subscripts
        return f"{cls.__name__}({sub!r}, shape={self.shape})"

    def sum(self, axis=None) -> np.ndarray:
        if isinstance(axis, int):
            axis = (axis,)

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
    def values(self) -> np.ndarray:
        return self.einsum(self.subscripts + "->" + self.subscripts)

    def __getitem__(self, index) -> "MatrixProduct":
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
        out = "".join(
            j for j in self.subscripts if not jnp.isscalar(sub_index.get(j)))
        return MatrixProduct(new_indices + "->" + out, *new_operands)

    def diagonal(self, subscripts: Optional[str] = None, axis1: Optional[str] = None, axis2: Optional[str] = None):
        subs: str = subscripts or self.subscripts
        ax1: str = axis1 or subs[-2]
        ax2: str = axis2 or subs[-1]

        def diagonals() -> Generator[Tuple[np.ndarray, str], None, None]:
            for M, sub in zip(self.operands, self.expand_subscript(subs)):
                if ax2 in sub:
                    if ax1 in sub:
                        M = M.diagonal(axis1=sub.index(
                            ax1), axis2=sub.index(ax2))
                        sub = sub.replace(ax2, "")
                    sub = sub.replace(ax2, ax1)
                yield M, sub

        new_operands, indices = zip(*diagonals())
        out = self.subscripts.replace(ax2, "")
        return MatrixProduct(",".join(indices) + "->" + out, *new_operands)


def func_jac(func, params, *args, **kwargs):
    l, vjp = jax.vjp(lambda p: func(p, *args, **kwargs), params)
    grad = vjp(1.)
    return l, grad


def make_func_jac(func):
    return partial(func_jac, func)


class OptMultiFunction:
    def __init__(self, func, unravel, args_groups, _sign=-1, concurrent_kws=None, **kwargs):
        self.func = func
        self._func_jac = make_func_jac(func)
        self.unravel = unravel
        self.args_groups = dict(args_groups)
        self.kwargs = kwargs
        self.sign = _sign
        self._grad = jax.grad(func)
        self.concurrent_kws = concurrent_kws or {}

    def map_concurrent(self, func, x, **kwargs):
        kwargs = {
            **self.concurrent_kws, **self.kwargs, **kwargs
        }
        params = self.unravel(x)
        res, errors = map_concurrent(
            func,
            {
                k: (params, *args)
                for k, args in self.args_groups.items()
            },
            **kwargs,
        )
        return res, errors

    def __call__(self, x, **kwargs):
        res, errors = self.map_concurrent(
            self.func, x, **kwargs
        )
        return res

    def func_jac(self, x, **kwargs):
        res, errors = self.map_concurrent(
            self._func_jac, x, **kwargs
        )
        return res


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
        return cls(transform.apply, unravel, *args, **kwargs, _sign=_sign)

    @classmethod
    def from_transform_and_optimize(cls, transform, *args, _sign=-1, _rng=None, min_kws=None, **kwargs):
        params = transform.init(_rng, *args, **kwargs)
        x, unravel = ravel_pytree(params)
        opt = cls(transform.apply, unravel, *args, **kwargs, _sign=_sign)
        res = opt.minimize(params, **(min_kws or {}))
        return opt, res

    @classmethod
    def transform_and_optimize(cls, func, *args, **kwargs):
        func_t = transform(func)
        return cls.from_transform_and_optimize(func_t, *args, **kwargs)

    def __call__(self, x):
        params = self.unravel(x)
        return self.sign * self.func(params, *self.args, **self.kwargs)

    def func_jac(self, x):
        params = self.unravel(x)
        l, vjp = jax.vjp(lambda p: self.func(
            p, *self.args, **self.kwargs), params)
        grad = vjp(1.)
        G = self.sign * self.ravel(grad)
        return self.sign * l, G

    def jac(self, x):
        params = self.unravel(x)
        grad = self._grad(params, *self.args, **self.kwargs)
        return self.sign * self.ravel(grad)

    @staticmethod
    def ravel(params):
        return np.array(ravel_pytree(params)[0], float)

    def minimize(self, params, jac=True, method='L-BFGS-B', **kwargs):
        from scipy import optimize

        x0 = self.ravel(params)
        if jac:
            res = optimize.minimize(
                self.func_jac, x0, method=method, jac=True, **kwargs)
        else:
            res = optimize.minimize(
                self, x0, method=method, jac=self.jac, **kwargs)

        res['params'] = self.unravel(res.x)
        res['gradient'] = self.unravel(res.jac)
        return res
