
from abc import ABC, abstractmethod
from math import prod

import numpy
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import haiku as hk

from .utils import to_2d

SQPI2 = jnp.sqrt(jnp.pi/2)
ISQ2 = jnp.sqrt(0.5) 


class AbstractKernel(ABC, hk.Module):
    @abstractmethod
    def k(self, X0, X1=None)-> numpy.ndarray:
        pass 

    def K(self, X0, X1=None) -> numpy.ndarray:
        X1 = X0 if X1 is None else X1
        X0, X1 = to_2d(X0, X1)
        return self.k(X0[:, None, :], X1[None, ...])

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

    @classmethod
    def with_bias(cls, *args, **kwargs) -> "SumKernel":
        return cls(*args, **kwargs) + Bias()


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
        X1 = X0 if X1 is None else X1
        X0, X1 = to_2d(X0, X1)
        return jnp.equal(X0, X1).all(axis=-1) * self.variance


class SEKernel(AbstractKernel):
    def __init__(self, scale=None, variance=None, *, name=None):
        super().__init__(name=name)
        self.variance = variance or jnp.exp(hk.get_parameter(
            "log_var", shape=(), dtype="f", init=jnp.zeros))
        self.scale = scale or jnp.exp(hk.get_parameter(
            "log_scale", shape=(), dtype="f", init=jnp.zeros))

    def k(self, X0, X1=None):
        X1 = X0 if X1 is None else X1
        X0, X1 = to_2d(X0, X1)
        return self.variance * se_kernel(
            X0, X1, self.scale
        )


class IntSEKernel(AbstractKernel):
    def __init__(self, t0=0., scale=None, variance=None, *, name=None, active_dim=0):
        super().__init__(name=name)
        self.t0 = t0
        self.active_dim = active_dim
        self.variance = variance or  jnp.exp(hk.get_parameter(
            "log_var", shape=(), dtype="f", init=jnp.zeros))
        self.scale = scale or  jnp.exp(hk.get_parameter(
            "log_scale", shape=(), dtype="f", init=jnp.zeros))

    def k(self, X1, X2=None):
        X1 = X2 if X1 is None else X1
        X1, X2 = to_2d(X1, X2)
        K = iint_se_kernel(
            self.t0, X1[..., self.active_dim], X2[..., self.active_dim], self.scale
        )
        return self.variance * K


class IntegralSEKernel(AbstractKernel):
    def __init__(self, scale=None, variance=None, bias=None, *, name=None, active_dim=0):
        super().__init__(name=name)
        self.active_dim = active_dim
        self.variance = variance or  jnp.exp(hk.get_parameter(
            "log_var", shape=(), dtype="f", init=jnp.zeros))
        self.scale = scale or  jnp.exp(hk.get_parameter(
            "log_scale", shape=(), dtype="f", init=jnp.zeros))
        self.bias = bias or  jnp.exp(hk.get_parameter(
            "bias", shape=(), dtype="f", init=jnp.zeros))

    def k(self, X1, X2=None):
        X1 = X2 if X1 is None else X1
        X1, X2 = to_2d(X1, X2)
        d12 = jnp.abs(
            X1[..., self.active_dim] - X2[..., self.active_dim]
        ) / self.scale
        k = self.variance * self.scale**2 * jnp.clip(
            1 + self.bias
            - jnp.exp( - jnp.square(d12)/2)
            - SQPI2 * d12 * jsp.special.erf(d12 * ISQ2),
            0, None
        )
        return k



def se_kernel(X1, X2, s=1.):
    d12 = (X1 - X2) / s 
    return jnp.exp(-jnp.square(d12).sum(-1) / 2)


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
