
from abc import ABC, abstractmethod
from math import prod
from typing import Callable, Optional

import numpy
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import haiku as hk

from scipy import integrate

from .utils import to_2d

SQPI2 = jnp.sqrt(jnp.pi/2)
ISQ2 = jnp.sqrt(0.5)


class AbstractKernel(ABC, hk.Module):
    @abstractmethod
    def k(self, X0, X1=None) -> numpy.ndarray:
        pass

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
    def with_bias(
            cls, *args, bias_name=None, bias_variance=None, **kwargs) -> "SumKernel":
        return cls(*args, **kwargs) + Bias(variance=bias_variance, name=bias_name)


class DotProduct(AbstractKernel):
    def __init__(self, variance=None, name=None, shape=()):
        super().__init__(name=name)
        self.variance = variance or jnp.exp(hk.get_parameter(
            "log_var", shape=shape, dtype="f", init=jnp.zeros))

    def k(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)
        return jax.vmap(jnp.dot)(X0 * self.variance, X1)

    def K(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)
        return (X0 * self.variance) @ X1.T


class ArcCosine(AbstractKernel):
    def __init__(
            self, variance=None, weight=None, bias=None, order=2, name=None, shape=()):
        super().__init__(name=name)
        self.variance = variance or jnp.exp(hk.get_parameter(
            "log_var", shape=(), dtype="f", init=jnp.zeros))
        self.weight = weight if weight is not None else jnp.exp(hk.get_parameter(
            "log_weight", shape=shape, dtype="f", init=jnp.zeros))
        self.bias = bias or jnp.exp(hk.get_parameter(
            "log_bias", shape=(), dtype="f", init=jnp.zeros))
        self.order = order

    def _weighted_dot(self, X0, X1):
        return jax.vmap(jnp.dot)(X0 * self.weight, X1) + self.bias

    def _J(self, theta):
        if self.order == 0:
            return jnp.pi - theta
        elif self.order == 1:
            return jnp.sin(theta) + (jnp.pi - theta) * jnp.cos(theta)
        else:
            return 3.0 * jnp.sin(theta) * jnp.cos(theta) + (jnp.pi - theta) * (
                1.0 + 2.0 * jnp.cos(theta) ** 2
            )

    def _kernel(self, x00, x11, x01):
        cos_theta = x01 / jnp.sqrt(x00 * x11)
        jitter = 1e-15  # improve numerical stability
        theta = jnp.arccos(jitter + (1 - 2 * jitter) * cos_theta)

        K = self._J(theta)
        K *= jnp.sqrt(x00) ** self.order
        K *= jnp.sqrt(x11) ** self.order
        K *= self.variance / jnp.pi
        return K

    def K(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)

        x00 = self._weighted_dot(X0, X0)[:, None]
        x11 = self._weighted_dot(X1, X1)[None, :]
        x01 = ((X0 * self.weight) @ X1.T) + self.bias
        return self._kernel(x00, x11, x01)

    def k(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)

        x00 = self._weighted_dot(X0, X0)
        x11 = self._weighted_dot(X1, X1)
        x01 = self._weighted_dot(X0, X1)

        return self._kernel(x00, x11, x01)


class Bias(AbstractKernel):
    def __init__(self, variance=None, name=None):
        super().__init__(name=name)
        self.variance = variance or jnp.exp(hk.get_parameter(
            "log_var", shape=(), dtype="f", init=jnp.zeros))

    def k(self, X0, X1=None):
        return jnp.full(len(X0), self.variance)

    def K(self, X0, X1=None):
        X1 = X0 if X1 is None else X1
        return jnp.full((len(X0), len(X1)), self.variance)


class SliceKernel(AbstractKernel):
    def __init__(self, kernel: AbstractKernel, active_dims, name=None):
        super().__init__(name=name)
        self.kernel = kernel
        self.active_dims = active_dims

    def slice_input(self, x: Optional[jax.Array]) -> Optional[jax.Array]:
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
        super().__init__(name=name)
        self.kernel = kernel
        self.exponent = exponent

    def k(self, X0, X1=None):
        return self.kernel.k(X0, X1)**self.exponent

    def K(self, X0, X1=None):
        return self.kernel.K(X0, X1)**self.exponent


class SumKernel(AbstractKernel):
    aggregate = sum

    def __init__(self, *kernels: AbstractKernel, name=None):
        super().__init__(name=name)
        self.kernels = kernels

    def k(self, X0, X1=None):
        return self.aggregate(k.k(X0, X1) for k in self.kernels)

    def K(self, X0, X1=None):
        return self.aggregate(k.K(X0, X1) for k in self.kernels)


class ProductKernel(SumKernel):
    aggregate = prod


class WhiteNoise(AbstractKernel):
    def __init__(self, variance=None, *, name=None):
        super().__init__(name=name)
        self.variance = variance or jnp.exp(hk.get_parameter(
            "log_var", shape=(), dtype="f", init=jnp.zeros))

    def k(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)
        return jnp.equal(X0, X1).all(axis=-1) * self.variance


class SEKernel(AbstractKernel):
    def __init__(self, scale=None, variance=None, *, name=None, shape=()):
        super().__init__(name=name)
        self.variance = variance or jnp.exp(hk.get_parameter(
            "log_var", shape=(), dtype="f", init=jnp.zeros))
        self.scale = scale if scale is not None else jnp.exp(hk.get_parameter(
            "log_scale", shape=shape, dtype="f", init=jnp.zeros))

    def k(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)
        return self.variance * se_kernel(X0, X1, self.scale)


def sqrt(x):
    pos = x > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 0)), 0)


def dist2d(X1, X2, s=1., axis=-1):
    return jnp.square((X1 - X2) / s).sum(axis)


class Matern(AbstractKernel):
    matern: Callable[[jax.Array], jax.Array]

    def __init__(self, scale=None, variance=None, *, name=None, shape=()):
        super().__init__(name=name)
        self.variance = variance or jnp.exp(hk.get_parameter(
            "log_var", shape=(), dtype="f", init=jnp.zeros))
        self.scale = scale if scale is not None else jnp.exp(hk.get_parameter(
            "log_scale", shape=shape, dtype="f", init=jnp.zeros))

    def dist2d(self, X0, X1=None):
        X0, X1 = self.to_2d(X0, X1)
        d01 = dist2d(X0, X1, self.scale)
        return sqrt(d01)

    def k(self, X0, X1=None):
        d = self.dist2d(X0, X1)
        return self.variance * self.matern(d)


class Matern12(Matern):
    @staticmethod
    def matern(d):
        return jnp.exp(- d)


class Matern32(Matern):
    @staticmethod
    def matern(d):
        sqrt3 = jnp.sqrt(3)
        return (1 + sqrt3 * d) * jnp.exp(- sqrt3 * d)


class Matern52(Matern):
    @staticmethod
    def matern(d):
        sqrt5 = jnp.sqrt(5)
        return (1 + sqrt5 * d + 5 / 3 * jnp.square(d)) * jnp.exp(- sqrt5 * d)


class IntSEKernel(AbstractKernel):
    def __init__(self, t0=0., scale=None, variance=None, *, name=None, active_dim=0):
        super().__init__(name=name)
        self.active_dim = active_dim

        self.t0 = t0 or hk.get_parameter(
            "t0", shape=(), dtype="f", init=jnp.zeros)
        self.variance = variance or jnp.exp(hk.get_parameter(
            "log_var", shape=(), dtype="f", init=jnp.zeros))
        self.scale = scale or jnp.exp(hk.get_parameter(
            "log_scale", shape=(), dtype="f", init=jnp.zeros))

    def k(self, X1, X2=None):
        X1 = X2 if X1 is None else X1
        X1, X2 = to_2d(X1, X2)
        K = iint_se_kernel(
            self.t0, X1[..., self.active_dim],
            X2[..., self.active_dim], self.scale
        )
        return self.variance * K


class IntegralSEKernel(AbstractKernel):
    def __init__(self, scale=None, variance=None, bias=None, *, name=None, active_dim=0):
        super().__init__(name=name)
        self.active_dim = active_dim
        self.variance = variance or jnp.exp(hk.get_parameter(
            "log_var", shape=(), dtype="f", init=jnp.zeros))
        self.scale = scale or jnp.exp(hk.get_parameter(
            "log_scale", shape=(), dtype="f", init=jnp.zeros))
        self.bias = bias or jnp.exp(hk.get_parameter(
            "bias", shape=(), dtype="f", init=jnp.zeros))

    def k(self, X1, X2=None):
        X1 = X2 if X1 is None else X1
        X1, X2 = to_2d(X1, X2)
        d12 = jnp.abs(
            X1[..., self.active_dim] - X2[..., self.active_dim]
        ) / self.scale
        k = self.variance * self.scale**2 * jnp.clip(
            1 + self.bias
            - jnp.exp(- jnp.square(d12)/2)
            - SQPI2 * d12 * jsp.special.erf(d12 * ISQ2),
            0, None
        )
        return k


def se_kernel(X1, X2, s=1., axis=-1):
    d12 = dist2d(X1, X2, s, axis=axis)
    return jnp.exp(-d12 / 2)


def nint_se_kernel(t0, t1, t2, s=1., with_err=False):
    val, err = integrate.quad(
        se_kernel,
        t0, t1,
        args=(t2, s),
    )
    if with_err:
        return val, err

    return val


def niint_se_kernel(t0, t1, t2, s=1., with_err=False):
    val, err = integrate.dblquad(
        se_kernel,
        t0, t1, t0, t2,
        args=(s,),
    )
    if with_err:
        return val, err

    return val


@jax.jit
def int_se_kernel(t0, t1, t2, s=1.):
    d12 = (t1 - t2) / s
    d02 = (t0 - t2) / s
    return SQPI2 * jnp.abs(
        jsp.special.erf(d12 * ISQ2)
        - jsp.special.erf(d02 * ISQ2)
    ) * s


@jax.jit
def iint_se_kernel(t0, t1, t2, s=1.):
    d12 = (t1 - t2) / s
    d01 = (t0 - t1) / s
    d02 = (t0 - t2) / s
    return (
        SQPI2 * (
            d01 * jsp.special.erf(d01 * ISQ2)
            + d02 * jsp.special.erf(d02 * ISQ2)
            - d12 * jsp.special.erf(d12 * ISQ2)
        )
        + jnp.exp(-d01**2 / 2)
        + jnp.exp(-d02**2 / 2)
        - jnp.exp(-d12**2 / 2)
        - 1
    ) * s**2
