
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import haiku as hk

from .gpr import GaussianProcessRegression
from .kernels import SEKernel, IntSEKernel, Constant, AbstractKernel
from .multi_kernel import AbstractMultiKernel, CovMultiKernel, DiagMultiKernel
from .utils import solve_triangular, transform 


class LinearGPCorrelatedRegression(GaussianProcessRegression):
    def __init__(self, X, W, y, *, t0=None, name=None, kernel=None, **kwargs):
        super().__init__(X, W, name=name, kernel=kernel, **kwargs)
        self.W = self.y
        self.y = jnp.asarray(y)

    def _set_kernel(self, kernel, **kwargs):
        self.kernel = kernel or CovMultiKernel(self.n_dims, SEKernel(**kwargs))

    def _gp_init(self):
        K = self.K()
        Kf = K.einsum("ijkl,ik,jl->ij", self.W, self.W) + jnp.eye(self.n_obs) * self.obs_var
        L = jsp.linalg.cho_factor(Kf)
        y = self.y - (self.W * self.mean(self.X)).sum(1)
        a = jsp.linalg.cho_solve(L, y)
        return Kf, L, y, a

    def predict_coef(self, X1):
        Kf, L, y, a = self._gp_init()
        k1 = self.k(X1)
        return k1.einsum("ijkl,jl,j->ik", self.W, a) + self.mean(X1)

    def __linear_gp_init(self):
        K = self.K()
        covW = self.W.dot(self.coef_cov)
        Kf = covW.dot(self.W.T) * K + jnp.eye(self.n_obs) * self.obs_var
        L = jsp.linalg.cho_factor(Kf)
        y = self.y - (self.W * self.mean(self.X)).sum(1)
        a = jsp.linalg.cho_solve(L, y)
        acovW = a[:, None] * covW
        return Kf, L, y, a, covW, acovW

    def predict(self, X1, W1):
        Z1 = self.predict_coef(X1)
        return (W1 * Z1).sum(1)

    def predict_coef_covar(self, X1):
        Kf, L, y, a = self._gp_init()
        k1 = self.k(X1)
        Z1 = k1.einsum("ijkl,jl,j->ik", self.W, a) + self.mean(X1)
        LWk1 = solve_triangular(
            L[0] if L[1] else L[0].T, k1.einsum("ijkl,jl->jik", self.W), lower=True
        )
        K11 = self.kernel.K(X1, X1)
        Z1_covar = (
            K11.einsum("ijkl") - jnp.einsum("ijk,ilm->jlkm", LWk1, LWk1)
        )
        return Z1, Z1_covar
        
    def predict_coef_var(self, X1):
        Kf, L, y, a = self._gp_init()
        k1 = self.k(X1)
        Z1 = k1.einsum("ijkl,jl,j->ik", self.W, a) + self.mean(X1)
        LWk1 = solve_triangular(
            L[0] if L[1] else L[0].T, k1.einsum("ijkl,jl->jik", self.W), lower=True
        )
        k11 = self.kernel.k(X1, X1)

        Z1_var = k11.diagonal().values - jnp.square(LWk1).sum(0)
        return Z1, Z1_var


def get_linear_gpr(times, X, observations, kernel=None, multi_kernel='cov', **kwargs):
    if kernel == 'integral':
        if multi_kernel == 'cov':
            kernel = CovMultiKernel(
                jnp.shape(X)[1], 
                kernel=IntSEKernel.with_constant()
            )
        elif multi_kernel == 'diag':
            kernel = DiagMultiKernel(
                [
                    IntSEKernel.with_constant() for i in range(jnp.shape(X)[1])
                ]
            )

    elif isinstance(kernel, AbstractKernel):
        kernel = CovMultiKernel(jnp.shape(X)[1], kernel=kernel)

    return LinearGPCorrelatedRegression(times, X, observations, kernel=kernel, **kwargs)


make_linear_gpr = transform(get_linear_gpr)


@transform
def linear_gpr_likelihood(times, X, observations, **kwargs):
    linear_gpr = get_linear_gpr(times, X, observations, **kwargs)
    return linear_gpr.log_likelihood()