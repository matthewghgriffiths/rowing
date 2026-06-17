from collections.abc import Callable
from math import prod

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy
from scipy import integrate

from .utils import to_2d

SQPI2 = jnp.sqrt(jnp.pi / 2)
ISQ2 = jnp.sqrt(0.5)


def _identity(x):
    return x


class Hyper(nnx.Module, pytree=False):
    """A kernel hyperparameter: either a fixed value or a learnable (log-)parameter.

    When ``value`` is given it is stored as a fixed constant (not optimised). Otherwise a
    learnable ``nnx.Param`` is created (initialised with ``init``), and ``.value`` returns
    ``transform`` applied to it (``jnp.exp`` by default, so the stored param is in log space).
    """

    def __init__(self, value=None, *, shape=(), init=jnp.zeros, transform=jnp.exp):
        self.transform = transform
        if value is None:
            self.param = nnx.Param(init(shape))
            self.fixed = None
        else:
            self.param = None
            self.fixed = value

    @property
    def value(self):
        if self.param is None:
            return self.fixed
        return self.transform(self.param[...])


class AbstractKernel(nnx.Module, pytree=False):
    def k(self, X0, X1=None) -> numpy.ndarray:
        raise NotImplementedError

    def K(self, X0, X1=None) -> numpy.ndarray:
        X0, X1 = self.to_2d(X0, X1)
        return self.k(X0[:, None, :], X1[None, ...])

    def to_2d(self, X0, X1=None):
        return to_2d(X0, X0 if X1 is None else X1)

    def __add__(self, other) -> "SumKernel":
        if isinstance(other, AbstractKernel):
            return SumKernel(self, other)
        elif jnp.isscalar(other):
            return SumKernel(self, Bias(other))
        else:
            raise ValueError(f"{other} incompatible")

    def __radd__(self, other) -> "SumKernel":
        return self + other

    def __mul__(self, other) -> "ProductKernel":
        if isinstance(other, AbstractKernel):
            return ProductKernel(self, other)
        elif jnp.isscalar(other):
            return ProductKernel(self, Bias(other))
        else:
            raise ValueError(f"{other} incompatible")

    def __pow__(self, exponent):
        return PowerKernel(self, exponent)

    def __getitem__(self, active_dims):
        return SliceKernel(self, active_dims)

    @classmethod
    def with_bias(cls, *args, bias_name=None, bias_variance=None, **kwargs) -> "SumKernel":
        return cls(*args, **kwargs) + Bias(variance=bias_variance, name=bias_name)


class DotProduct(AbstractKernel):
    def __init__(self, offset=None, variance=None, name=None, shape=()):
        self.offset = Hyper(offset, shape=shape, transform=_identity)
        self.variance = Hyper(variance, shape=shape)

    def k(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)
        return jax.vmap(jnp.dot)((X0 - self.offset.value) * self.variance.value, (X1 - self.offset.value))

    def K(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)
        return (X0 * self.variance.value) @ X1.T


class ChangePoint(AbstractKernel):
    def __init__(self, change=None, scale=None, variance=None, name=None, shape=()):
        self.change = Hyper(change, shape=shape, init=jnp.zeros, transform=_identity)
        self.scale = Hyper(scale, shape=shape, init=jnp.ones, transform=_identity)
        self.variance = Hyper(variance, shape=shape)

    def k(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)
        S0 = jax.nn.sigmoid((X0 - self.change.value) / self.scale.value)
        S1 = jax.nn.sigmoid((X1 - self.change.value) / self.scale.value)
        return (S0 * self.variance.value * S1).sum(-1)


class ChangePoints(AbstractKernel):
    def __init__(self, change, scale=None, variance=None, name=None):
        self.change = jnp.sort(change)
        self.scale = Hyper(scale, shape=self.change.shape, init=jnp.ones, transform=_identity)
        self.variance = Hyper(variance, shape=self.change[1:].shape)

    def k(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)
        S0 = jax.nn.sigmoid((X0 - self.change) / self.scale.value)
        S1 = jax.nn.sigmoid((X1 - self.change) / self.scale.value)
        R0 = S0[..., :-1] * (1 - S0[..., 1:])
        R1 = S1[..., :-1] * (1 - S1[..., 1:])
        return (R0 * R1 * self.variance.value).sum(-1)


class ArcCosine(AbstractKernel):
    def __init__(self, variance=None, weight=None, bias=None, order=2, name=None, shape=()):
        self.variance = Hyper(variance)
        self.weight = Hyper(weight, shape=shape)
        self.bias = Hyper(bias)
        self.order = order

    def _weighted_dot(self, X0, X1):
        return jax.vmap(jnp.dot)(X0 * self.weight.value, X1) + self.bias.value

    def _J(self, theta):
        if self.order == 0:
            return jnp.pi - theta
        elif self.order == 1:
            return jnp.sin(theta) + (jnp.pi - theta) * jnp.cos(theta)
        else:
            return 3.0 * jnp.sin(theta) * jnp.cos(theta) + (jnp.pi - theta) * (1.0 + 2.0 * jnp.cos(theta) ** 2)

    def _kernel(self, x00, x11, x01):
        cos_theta = x01 / jnp.sqrt(x00 * x11)
        jitter = 1e-15  # improve numerical stability
        theta = jnp.arccos(jitter + (1 - 2 * jitter) * cos_theta)

        K = self._J(theta)
        K *= jnp.sqrt(x00) ** self.order
        K *= jnp.sqrt(x11) ** self.order
        K *= self.variance.value / jnp.pi
        return K

    def K(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)

        x00 = self._weighted_dot(X0, X0)[:, None]
        x11 = self._weighted_dot(X1, X1)[None, :]
        x01 = ((X0 * self.weight.value) @ X1.T) + self.bias.value
        return self._kernel(x00, x11, x01)

    def k(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)

        x00 = self._weighted_dot(X0, X0)
        x11 = self._weighted_dot(X1, X1)
        x01 = self._weighted_dot(X0, X1)

        return self._kernel(x00, x11, x01)


class Bias(AbstractKernel):
    def __init__(self, variance=None, name=None):
        self.variance = Hyper(variance)

    def k(self, X0, X1=None):
        return jnp.full(len(X0), self.variance.value)

    def K(self, X0, X1=None):
        X1 = X0 if X1 is None else X1
        return jnp.full((len(X0), len(X1)), self.variance.value)


class SliceKernel(AbstractKernel):
    def __init__(self, kernel: AbstractKernel, active_dims, name=None):
        self.kernel = kernel
        self.active_dims = active_dims

    def slice_input(self, x: jax.Array | None) -> jax.Array | None:
        if x is None:
            return

        return x[..., self.active_dims]

    def k(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)
        return self.kernel.k(self.slice_input(X0), self.slice_input(X1))

    def K(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)
        return self.kernel.K(self.slice_input(X0), self.slice_input(X1))


class PowerKernel(AbstractKernel):
    def __init__(self, kernel: AbstractKernel, exponent, name=None):
        self.kernel = kernel
        self.exponent = exponent

    def k(self, X0, X1=None):
        return self.kernel.k(X0, X1) ** self.exponent

    def K(self, X0, X1=None):
        return self.kernel.K(X0, X1) ** self.exponent


class SumKernel(AbstractKernel):
    aggregate = staticmethod(sum)

    def __init__(self, *kernels: AbstractKernel, name=None):
        self.kernels = nnx.data(list(kernels))

    def k(self, X0, X1=None):
        return self.aggregate(k.k(X0, X1) for k in self.kernels)

    def K(self, X0, X1=None):
        return self.aggregate(k.K(X0, X1) for k in self.kernels)


class ProductKernel(SumKernel):
    aggregate = staticmethod(prod)


class WhiteNoise(AbstractKernel):
    def __init__(self, variance=None, *, name=None):
        self.variance = Hyper(variance)

    def k(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)
        return jnp.equal(X0, X1).all(axis=-1) * self.variance.value


class SEKernel(AbstractKernel):
    def __init__(self, scale=None, variance=None, *, name=None, shape=()):
        self.variance = Hyper(variance)
        self.scale = Hyper(scale, shape=shape)

    def k(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)
        return self.variance.value * se_kernel(X0, X1, self.scale.value)


def sqrt(x):
    pos = x > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 0)), 0)


def dist2d(X1, X2, s=1.0, axis=-1):
    return jnp.square((X1 - X2) / s).sum(axis)


class Matern(AbstractKernel):
    matern: Callable[[jax.Array], jax.Array]

    def __init__(self, scale=None, variance=None, *, name=None, shape=()):
        self.variance = Hyper(variance)
        self.scale = Hyper(scale, shape=shape)

    def dist2d(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)
        d01 = dist2d(X0, X1, self.scale.value)
        return sqrt(d01)

    def k(self, X0, X1=None):
        d = self.dist2d(X0, X1)
        return self.variance.value * self.matern(d)


def matern12(d):
    return jnp.exp(-d)


def matern32(d):
    sqrt3 = jnp.sqrt(3)
    return (1 + sqrt3 * d) * jnp.exp(-sqrt3 * d)


def matern52(d):
    sqrt5 = jnp.sqrt(5)
    return (1 + sqrt5 * d + 5 / 3 * jnp.square(d)) * jnp.exp(-sqrt5 * d)


class Matern12(Matern):
    matern = staticmethod(matern12)


class Matern32(Matern):
    matern = staticmethod(matern32)


class Matern52(Matern):
    matern = staticmethod(matern52)


class IntSEKernel(AbstractKernel):
    def __init__(self, t0=0.0, scale=None, variance=None, *, name=None, active_dim=0):
        self.active_dim = active_dim
        self.t0 = Hyper(t0 or None, init=jnp.zeros, transform=_identity)
        self.variance = Hyper(variance)
        self.scale = Hyper(scale)

    def k(self, X1, X2=None):
        X1 = X2 if X1 is None else X1
        X1, X2 = to_2d(X1, X2)
        K = iint_se_kernel(self.t0.value, X1[..., self.active_dim], X2[..., self.active_dim], self.scale.value)
        return self.variance.value * K


class IntegralSEKernel(AbstractKernel):
    def __init__(self, scale=None, variance=None, bias=None, *, name=None, active_dim=0):
        self.active_dim = active_dim
        self.variance = Hyper(variance)
        self.scale = Hyper(scale)
        self.bias = Hyper(bias)

    def k(self, X1, X2=None):
        X1 = X2 if X1 is None else X1
        X1, X2 = to_2d(X1, X2)
        d12 = jnp.abs(X1[..., self.active_dim] - X2[..., self.active_dim]) / self.scale.value
        k = (
            self.variance.value
            * self.scale.value**2
            * jnp.clip(1 + self.bias.value - jnp.exp(-jnp.square(d12) / 2) - SQPI2 * d12 * jsp.special.erf(d12 * ISQ2), 0, None)
        )
        return k


def se_kernel(X1, X2, s=1.0, axis=-1):
    d12 = dist2d(X1, X2, s, axis=axis)
    return jnp.exp(-d12 / 2)


def nint_se_kernel(t0, t1, t2, s=1.0, with_err=False):
    val, err = integrate.quad(
        se_kernel,
        t0,
        t1,
        args=(t2, s),
    )
    if with_err:
        return val, err

    return val


def niint_se_kernel(t0, t1, t2, s=1.0, with_err=False):
    val, err = integrate.dblquad(
        se_kernel,
        t0,
        t1,
        t0,
        t2,
        args=(s,),
    )
    if with_err:
        return val, err

    return val


@jax.jit
def int_se_kernel(t0, t1, t2, s=1.0):
    d12 = (t1 - t2) / s
    d02 = (t0 - t2) / s
    return SQPI2 * jnp.abs(jsp.special.erf(d12 * ISQ2) - jsp.special.erf(d02 * ISQ2)) * s


@jax.jit
def iint_se_kernel(t0, t1, t2, s=1.0):
    d12 = (t1 - t2) / s
    d01 = (t0 - t1) / s
    d02 = (t0 - t2) / s
    return (
        SQPI2 * (d01 * jsp.special.erf(d01 * ISQ2) + d02 * jsp.special.erf(d02 * ISQ2) - d12 * jsp.special.erf(d12 * ISQ2))
        + jnp.exp(-(d01**2) / 2)
        + jnp.exp(-(d02**2) / 2)
        - jnp.exp(-(d12**2) / 2)
        - 1
    ) * s**2


def sin_dist2d(X1, X2, period=1.0, axis=-1):
    return jnp.abs(jnp.sin(jnp.sqrt(dist2d(X1, X2, period, axis=axis)) * jnp.pi))


def se_periodic_kernel(X1, X2, period=1.0, l=1.0, axis=-1):  # noqa: E741
    d12 = sin_dist2d(X1, X2, period, axis=axis) / l
    return jnp.exp(-2 * d12**2)


class SEPeriodicKernel(AbstractKernel):
    def __init__(self, period=None, scale=None, variance=None, *, name=None, shape=()):
        self.variance = Hyper(variance)
        self.period = Hyper(period, shape=shape)
        self.scale = Hyper(scale, shape=shape)

    def k(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)
        return self.variance.value * se_periodic_kernel(X0, X1, self.period.value, self.scale.value)


class PeriodicMatern(Matern):
    matern: Callable[[jax.Array], jax.Array]

    def __init__(self, period=None, scale=None, variance=None, *, name=None, shape=()):
        self.variance = Hyper(variance)
        self.period = Hyper(period, shape=shape)
        self.scale = Hyper(scale, shape=shape)

    def dist2d(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)
        return sin_dist2d(X0, X1, self.period.value) / self.scale.value


class PeriodicMatern12(PeriodicMatern):
    matern = staticmethod(matern52)


class PeriodicMatern32(PeriodicMatern):
    matern = staticmethod(matern32)
