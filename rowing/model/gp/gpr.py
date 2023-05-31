
import numpy
import jax.numpy as jnp
import jax.scipy as jsp
import haiku as hk

from .utils import transform, solve_triangular, to_2d
from .kernels import SEKernel, IntSEKernel


class GaussianProcessRegression(hk.Module):
    def __init__(
            self, X, y, *, kernel=None, mean=None, obs_var=None, name=None, 
            kernel_kws=None
    ):
        super().__init__(name=name)
        self.X = to_2d(X)
        self.y = jnp.asarray(y)
        self._init()

        self._set_mean(mean)
        self._set_kernel(kernel, **(kernel_kws or {}))
            
        if obs_var is None:
            obs_var = jnp.eye(self.n_obs) * jnp.exp(hk.get_parameter(
                "obs_var", shape=(), dtype="f", init=jnp.zeros))

        self.obs_var = obs_var

    def _init(self):
        self.n_obs = len(self.X)
        self.obs_shape = self.y.shape
        self.dims = self.obs_shape[1:]
        self.n_dims = self.dims[0] if self.dims else 1
        self.size = self.y.size

    def _set_kernel(self, kernel, **kwargs):
        self.kernel = kernel or SEKernel(**kwargs)

    def _set_mean(self, mean):
        if isinstance(mean, numpy.ndarray):
            self.mean_val = jnp.reshape(mean, (1,) + self.dims)
        elif jnp.isscalar(mean):
            self.mean_val = jnp.full((1,) + self.dims, mean)
        elif mean:
            self.mean = mean 
        else:
            self.mean_val = hk.get_parameter(
                "gp_mean", shape=self.dims, dtype="f", init=jnp.zeros
            ).reshape((1,) + self.dims)

    def mean(self, t):
        return jnp.repeat(self.mean_val, len(t), axis=0)

    def _gp_init(self):
        K = self.K() + self.obs_var
        L = jsp.linalg.cho_factor(K)
        y = self.y - self.mean(self.X)
        a = jsp.linalg.cho_solve(L, y)
        return K, L, y, a

    def K(self):
        return self.kernel.K(self.X)

    def k(self, X1):
        return self.kernel.K(X1, self.X)

    def predict(self, X1):
        K, L, y, a = self._gp_init()
        return self.k(X1).dot(a) + self.mean(X1)

    def predict_var(self, X1):
        K, L, y, a = self._gp_init()

        k1 = self.k(X1)
        pred = k1.dot(a) + self.mean(X1)

        Lk1 = jsp.linalg.solve_triangular(
            L[0] if L[1] else L[0].T, k1.T, lower=True
        )
        pred_var = self.kernel.k(X1, X1) - jnp.square(Lk1).sum(0)
        
        return pred, pred_var

    def predict_covar(self, X1):
        K, L, y, a = self._gp_init()

        k1 = self.k(X1)
        pred = k1.dot(a) + self.mean(X1)

        Lk1 = jsp.linalg.solve_triangular(
            L[0] if L[1] else L[0].T, k1.T, lower=True
        )
        pred_covar = self.kernel.K(X1, X1) - Lk1.T.dot(Lk1)
        
        return pred, pred_covar

    def log_likelihood(self):
        K, L, y, a = self._gp_init()

        return - (a * y).sum() / 2 - L[0].diagonal().sum() * self.n_dims - self.size * jnp.log(2 * jnp.pi)


class LinearGPCorrelatedRegression(GaussianProcessRegression):
    def __init__(self, t, W, y, *, t0=None, name=None, kernel=None, mean=None, coef_cov=None):
        super().__init__(t, y, name=name, kernel=kernel, mean=mean)
        self.W = jnp.asarray(W)
        self.n_coef = self.W.shape[1]

        if coef_cov is None:
            coef_cov = jnp.diag(jnp.exp(hk.get_parameter(
                "coef_var", shape=(self.n_coef,), dtype="f", init=jnp.zeros
            )))
        
        self.coef_cov = coef_cov

    def mean(self, t):
        return jnp.zeros((len(t), self.n_coef), 'f')

    def _linear_gp_init(self):
        K = self.K()
        covW = self.W.dot(self.coef_cov)
        Kf = covW.dot(self.W.T) * K + jnp.eye(self.n_obs) * self.obs_var
        L = jsp.linalg.cho_factor(Kf)
        y = self.y - (self.W * self.mean(self.X)).sum(1)
        a = jsp.linalg.cho_solve(L, y)
        acovW = a[:, None] * covW
        return Kf, L, y, a, covW, acovW

    def _gp_init(self):
        Kf, L, y, a, covW, acovW = self._linear_gp_init()
        return Kf, L, y, a

    def predict(self, X1, W1):
        Z1 = self.predict_coef(X1)
        return (W1 * Z1).sum(1)

    def predict_coef(self, X1):
        Kf, L, y, a, covW, acovW = self._linear_gp_init()
        k1 = self.k(X1)
        return k1.dot(acovW) + self.mean(X1)

    def predict_coef_covar(self, X1):
        Kf, L, y, a, covW, acovW = self._linear_gp_init()
        k1 = self.k(X1)
        Z1 = k1.dot(acovW) + self.mean(X1)
        
        LcovWk1 = solve_triangular(
            L[0] if L[1] else L[0].T, 
            (covW[..., None] * k1.T[:, None, :]), 
            lower=True
        )
        K11 = self.kernel.K(X1, X1)
        Z1_covar = (
            (self.coef_cov[..., None, None] * K11[None, None, ...])
            - jnp.einsum("ijk,ilm->jlkm", LcovWk1, LcovWk1)
        )        
        return Z1, Z1_covar
        
    def predict_coef_var(self, X1):
        Kf, L, y, a, covW, acovW = self._linear_gp_init()
        k1 = self.k(X1)
        Z1 = k1.dot(acovW) + self.mean(X1)

        LcovWk1 = solve_triangular(
            L[0] if L[1] else L[0].T, (covW[:, None, :] * k1.T[..., None]), lower=True
        )
        k11 = self.kernel.k(X1, X1)

        Z1_var = (
            (self.coef_cov.diagonal()[None, :] * k11[:, None])
            - jnp.square(LcovWk1).sum(0)
        )
        return Z1, Z1_var


def get_gpr(times, observations, **kwargs):
    if 'kernel' not in kwargs:
        kwargs['kernel'] = IntSEKernel(times[0])
    return GaussianProcessRegression(times, observations, **kwargs)


make_gpr = transform(get_gpr)


@transform
def gpr_likelihood(times, observations, **kwargs):
    gp = get_gp(times, observations, **kwargs)
    return gp.log_likelihood()

