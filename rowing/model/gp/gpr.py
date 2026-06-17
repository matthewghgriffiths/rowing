import flax.nnx as nnx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy

from .kernels import Hyper, IntSEKernel, SEKernel, _identity
from .utils import solve_triangular, to_2d


class GaussianProcessRegression(nnx.Module, pytree=False):
    def __init__(self, X, y, *, kernel=None, mean=None, obs_var=None, name=None, kernel_kws=None):
        self.X = to_2d(X)
        self.y = jnp.asarray(y)

        self.n_obs = len(self.X)
        self.obs_shape = self.y.shape
        self.dims = self.obs_shape[1:]
        self.n_dims = self.dims[0] if self.dims else 1
        self.size = self.y.size

        self._set_mean(mean)
        self._set_kernel(kernel, **(kernel_kws or {}))

        if obs_var is None:
            self._obs_var_h = Hyper(None, init=jnp.zeros, transform=jnp.exp)
            self._obs_var_fixed = None
        else:
            self._obs_var_h = None
            self._obs_var_fixed = obs_var

    @property
    def obs_var(self):
        if self._obs_var_fixed is not None:
            return self._obs_var_fixed
        return jnp.eye(self.n_obs) * self._obs_var_h.value

    def _set_kernel(self, kernel, **kwargs):
        self.kernel = kernel if kernel is not None else SEKernel(**kwargs)

    def _set_mean(self, mean):
        self._mean_fn = None
        self._mean_const = None
        self.gp_mean = None
        if isinstance(mean, numpy.ndarray):
            self._mean_const = jnp.reshape(mean, (1,) + self.dims)
        elif jnp.isscalar(mean):
            self._mean_const = jnp.full((1,) + self.dims, mean)
        elif mean:
            self._mean_fn = mean
        else:
            self.gp_mean = Hyper(None, shape=self.dims, init=jnp.zeros, transform=_identity)

    def mean(self, t):
        if self._mean_fn is not None:
            return self._mean_fn(t)
        if self.gp_mean is not None:
            mean_val = self.gp_mean.value.reshape((1,) + self.dims)
        else:
            mean_val = self._mean_const
        return jnp.repeat(mean_val, len(t), axis=0)

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

        Lk1 = jsp.linalg.solve_triangular(L[0] if L[1] else L[0].T, k1.T, lower=True)
        pred_var = self.kernel.k(X1, X1) - jnp.square(Lk1).sum(axis=0)

        return pred, pred_var

    def predict_covar(self, X1):
        K, L, y, a = self._gp_init()

        k1 = self.k(X1)
        pred = k1.dot(a) + self.mean(X1)

        Lk1 = jsp.linalg.solve_triangular(L[0] if L[1] else L[0].T, k1.T, lower=True)
        pred_covar = self.kernel.K(X1, X1) - Lk1.T.dot(Lk1)

        return pred, pred_covar

    def log_likelihood(self):
        K, L, y, a = self._gp_init()

        return -(a * y).sum() / 2 - L[0].diagonal().sum() * self.n_dims - self.size * jnp.log(2 * jnp.pi)


class LinearGPCorrelatedRegression(GaussianProcessRegression):
    def __init__(self, t, W, y, *, t0=None, name=None, kernel=None, mean=None, coef_cov=None):
        super().__init__(t, y, name=name, kernel=kernel, mean=mean)
        self.W = jnp.asarray(W)
        self.n_coef = self.W.shape[1]

        if coef_cov is None:
            self._coef_cov_h = Hyper(None, shape=(self.n_coef,), transform=jnp.exp)
            self._coef_cov_fixed = None
        else:
            self._coef_cov_h = None
            self._coef_cov_fixed = coef_cov

    @property
    def coef_cov(self):
        if self._coef_cov_fixed is not None:
            return self._coef_cov_fixed
        return jnp.diag(self._coef_cov_h.value)

    def mean(self, t):
        return jnp.zeros((len(t), self.n_coef), "f")

    def _linear_gp_init(self):
        K = self.K()
        covW = self.W.dot(self.coef_cov)
        Kf = covW.dot(self.W.T) * K + jnp.eye(self.n_obs) * self.obs_var
        L = jsp.linalg.cho_factor(Kf)
        y = self.y - (self.W * self.mean(self.X)).sum(axis=1)
        a = jsp.linalg.cho_solve(L, y)
        acovW = a[:, None] * covW
        return Kf, L, y, a, covW, acovW

    def _gp_init(self):
        Kf, L, y, a, covW, acovW = self._linear_gp_init()
        return Kf, L, y, a

    def predict(self, X1, W1):
        Z1 = self.predict_coef(X1)
        return (W1 * Z1).sum(axis=1)

    def predict_coef(self, X1):
        Kf, L, y, a, covW, acovW = self._linear_gp_init()
        k1 = self.k(X1)
        return k1.dot(acovW) + self.mean(X1)

    def predict_coef_covar(self, X1):
        Kf, L, y, a, covW, acovW = self._linear_gp_init()
        k1 = self.k(X1)
        Z1 = k1.dot(acovW) + self.mean(X1)

        LcovWk1 = solve_triangular(L[0] if L[1] else L[0].T, (covW[..., None] * k1.T[:, None, :]), lower=True)
        K11 = self.kernel.K(X1, X1)
        Z1_covar = (self.coef_cov[..., None, None] * K11[None, None, ...]) - jnp.einsum("ijk,ilm->jlkm", LcovWk1, LcovWk1)
        return Z1, Z1_covar

    def predict_coef_var(self, X1):
        Kf, L, y, a, covW, acovW = self._linear_gp_init()
        k1 = self.k(X1)
        Z1 = k1.dot(acovW) + self.mean(X1)

        LcovWk1 = solve_triangular(L[0] if L[1] else L[0].T, (covW[:, None, :] * k1.T[..., None]), lower=True)
        k11 = self.kernel.k(X1, X1)

        Z1_var = (self.coef_cov.diagonal()[None, :] * k11[:, None]) - jnp.square(LcovWk1).sum(axis=0)
        return Z1, Z1_var


def get_gpr(times, observations, **kwargs):
    if "kernel" not in kwargs:
        kwargs["kernel"] = IntSEKernel(times[0])
    return GaussianProcessRegression(times, observations, **kwargs)
