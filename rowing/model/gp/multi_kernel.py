
from abc import ABC, abstractmethod

import jax.numpy as jnp
import haiku as hk

from .utils import MatrixProduct, get_pos_def, to_2d


class AbstractMultiKernel(ABC, hk.Module):
    @abstractmethod
    def k(self, X0, X1)-> MatrixProduct:
        pass 

    @abstractmethod
    def K(self, X0, X1) -> MatrixProduct:
        pass


class CovMultiKernel(AbstractMultiKernel):
    def __init__(self, n_output, kernel, *, coef_cov=None, dof=0, name=None):
        super().__init__(name=name)
        self.n_output = n_output
        self.kernel = kernel

        if coef_cov is None:
            coef_cov = get_pos_def(
                self.n_output, dof=dof, name="cov_kernel"
            )
        self.coef_cov = coef_cov

    def k(self, X0, X1=None):
        X1 = X0 if X1 is None else X1
        return MatrixProduct("i,jk", self.kernel.k(X0, X1), self.coef_cov)

    def K(self, X0, X1=None):
        X1 = X0 if X1 is None else X1
        X0, X1 = to_2d(X0, X1)
        return MatrixProduct("ij,kl", self.kernel.K(X0, X1), self.coef_cov)


class DiagMultiKernel(AbstractMultiKernel):
    def __init__(self, kernels, name=None):
        super().__init__(name=name)
        self.kernels = list(kernels)
        self.n_output = len(self.kernels) 
    
    def k(self, X0, X1=None):
        X1 = X0 if X1 is None else X1
        k = jnp.array([kernel.k(X0, X1) for kernel in self.kernels])
        return MatrixProduct("ij,ik->jik", k, jnp.eye(self.n_output))

    def K(self, X0, X1=None):
        X1 = X0 if X1 is None else X1
        X0, X1 = to_2d(X0, X1)
        K = jnp.array([kernel.K(X0, X1) for kernel in self.kernels])
        return MatrixProduct("kij,kl->ijkl", K, jnp.eye(self.n_output))

class CorrMultiKernel(DiagMultiKernel):
    def __init__(self, kernels, name=None):
        pass