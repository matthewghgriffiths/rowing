
from abc import ABC, abstractmethod
from math import prod

import numpy
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree
import haiku as hk

from .utils import to_2d, MatrixProduct

SQPI2 = jnp.sqrt(jnp.pi/2)
ISQ2 = jnp.sqrt(0.5) 


class AbstractKernel(ABC, hk.Module):
    @abstractmethod
    def k(self, X0, X1)-> numpy.ndarray:
        pass 

    @abstractmethod
    def K(self, X0, X1) -> numpy.ndarray:
        pass

    def __add__(self, other):
        if isinstance(other, AbstractKernel):
            return AdditionKernel(self, other) 
        elif jnp.isscalar(other):
            return AdditionKernel(self, Constant(other))
        else:
            raise ValueError(f"{other} incompatible")

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, AbstractKernel):
            return AdditionKernel(self, other) 
        elif jnp.isscalar(other):
            return AdditionKernel(self, Constant(other))

    @classmethod
    def with_constant(cls, *args, **kwargs):
        return cls(*args, **kwargs) + Constant()


class Constant(AbstractKernel):
    def __init__(self, constant=None, name=None):
        super().__init__(name=name)
        self.constant = constant or hk.get_parameter(
            "constant", shape=(), dtype="f", init=jnp.ones)

    def k(self, X0, X1):
        return jnp.full(len(X0), self.constant)

    def K(self, X0, X1):
        return jnp.full((len(X0), len(X1)), self.constant)


class AdditionKernel(AbstractKernel):
    def __init__(self, *kernels, name=None):
        super().__init__(name=name)
        self.kernels = kernels

    def k(self, X0, X1):
        return sum(k.k(X0, X1) for k in self.kernels)

    def K(self, X0, X1):
        return sum(k.K(X0, X1) for k in self.kernels)


class ProductKernel(AbstractKernel):
    def __init__(self, *kernels, name=None):
        super().__init__(name=name)
        self.kernels = kernels

    def k(self, X0, X1):
        return prod(k.k(X0, X1) for k in self.kernels)

    def K(self, X0, X1):
        return prod(k.K(X0, X1) for k in self.kernels)

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

    def K(self, X0, X1=None):
        X1 = X0 if X1 is None else X1
        X0, X1 = to_2d(X0, X1)
        return self.k(X0[:, None, :], X1[None, ...])


class IntSEKernel(AbstractKernel):
    def __init__(self, t0=0., *, name=None, col=0):
        super().__init__(name=name)
        self.t0 = t0
        self.col = col
        self.variance = jnp.exp(hk.get_parameter(
            "log_var", shape=(), dtype="f", init=jnp.zeros))
        self.scale = jnp.exp(hk.get_parameter(
            "log_scale", shape=(), dtype="f", init=jnp.zeros))
        # self.const = hk.get_parameter(
        #     "const", shape=(), dtype="f", init=jnp.ones)

    def k(self, X1, X2=None):
        X1 = X0 if X1 is None else X1
        X1, X2 = to_2d(X1, X2)
        K = iint_se_kernel(
            self.t0, X1[..., self.col], X2[..., self.col], self.scale
        )
        return self.variance * K

    def K(self, X1, X2=None):
        X2 = X1 if X2 is None else X2
        X1, X2 = to_2d(X1, X2)
        return self.k(X1[:, None, :], X2[None, ...])


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
    return SQPI2 * (
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
