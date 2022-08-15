
import numpy as np

import jax 
import jax.numpy as jnp
from tensorflow_probability.substrates.jax.math import cholesky_update

def estimate_orientation(accel_data, gyro_data):
    z = accel_data.mean()
    z /= np.linalg.norm(z) 
    
    gyro_cov = gyro_data.cov()
    x = ((1 - np.eye(3).dot(z)) * np.eye(3)).sum(1)
    for i in range(5):
        x = gyro_cov.values.dot(x)
        x -= x.dot(z) * z
        x /= np.linalg.norm(x)

    y = np.cross(z, x)

    return np.array([x, y, z])

@jax.jit
def rowing_stroke_model(dt, n, s, v0, cs, ds):
    cs = jnp.array(cs)
    ds = jnp.array(ds)
    j = jnp.arange(1, len(cs) + 1) * jnp.pi * 2

    cj, dj = (cs.T/j).T, (ds.T/j).T
    cjj, djj = (cj.T/j).T, (dj.T/j).T
    
    ns = (n.T * j).T 
    n1 = n + s * dt
    n1s = (n1.T * j).T 
    
    cosn = jnp.cos(ns)
    sinn = jnp.sin(ns)
    cosn1 = jnp.cos(n1s)
    sinn1 = jnp.sin(n1s)

    a1 = (cs * cosn + ds * sinn).sum(0)
    v1 = v0 + (cj * sinn - dj * cosn).sum(0) / s
    dx = v0 * dt + (
        cjj * cosn + djj * sinn
        - cjj * cosn1 - djj * sinn1
    ).sum(0) / s / s

    return n1, dx, v1, a1

@jax.jit 
def follow_bearing(lat, lon, bearing, distance, radius=1):
    d = distance / radius
    lat1 = jnp.arcsin(
        jnp.sin(lat)*jnp.cos(d) + jnp.cos(lat)*jnp.sin(d)*jnp.cos(bearing))
    lon1 = lon + jnp.arctan2(
        jnp.sin(bearing)*jnp.sin(d)*jnp.cos(lat), 
        jnp.cos(d)-jnp.sin(lat)*jnp.sin(lat1)
    )
    return lat1, lon1

@jax.jit
def euler2mat(vec):
    s_1, s_2, s_3 = jnp.sin(vec)
    c_1, c_2, c_3 = jnp.cos(vec)
    return jnp.array([
        [c_1 * c_3 - c_2 * s_1 * s_3, -c_1 * s_3 - c_2 * c_3 * s_1, s_1 * s_2], 
        [c_3 * s_1+c_1 * c_2 * s_3, c_1 * c_2 * c_3-s_1 * s_3, -c_1 * s_2],
        [s_2 * s_3, c_3 * s_2, c_2]
    ])


def _exp_rolling_lombscargle(carry, xt):
    AAT, Ax, t0, freqs, harms, alpha = carry
    n_freqs = len(freqs)
    t, x = xt
    dt = t - t0
    e1 = 1 - dt * alpha
    
    a = jnp.c_[
        jnp.ones((n_freqs, 1)), 
        jnp.sin(freqs[:, None] * harms[None, :] * t), 
        jnp.cos(freqs[:, None] * harms[None, :] * t),
    ]
    AAT1 = AAT * e1 + a[:, None, :] * a[:, :, None]
    Ax1 = Ax * e1 + a * x
    coefs = jnp.linalg.solve(AAT1, Ax1)
    return (AAT1, Ax1, t, freqs, harms, alpha), coefs


def exp_rolling_lombscargle(times, acceleration, freqs, n_harms, alpha=1e-1):
    n_freqs = len(freqs)
    harms = np.arange(1, n_harms + 1)
    xt = jnp.c_[times, acceleration]
    AAT = jnp.concatenate([
        jnp.eye(2 * n_harms + 1)[None, ...] for i in range(n_freqs)
    ])
    Ax = jnp.zeros((n_freqs, 2 * n_harms + 1))
    init = (AAT, Ax, times[0], freqs, harms, alpha)
    carry, coefs = jax.lax.scan(
        _exp_rolling_lombscargle, init, xt
    )
    return coefs


def _exp_rolling_lombscargle_cho(carry, xt):
    AAT, Ax, t0, freqs, harms, alpha = carry
    n_freqs = len(freqs)
    t, x = xt
    dt = t - t0
    e1 = 1 - dt * alpha
    
    a = jnp.c_[
        jnp.ones((n_freqs, 1)), 
        jnp.cos(freqs[:, None] * harms[None, :] * t), 
        jnp.sin(freqs[:, None] * harms[None, ::-1] * t),
    ]
    AAT1 = cholesky_update(AAT * jnp.sqrt(e1), a)
    Ax1 = Ax * e1 + a * x
    coefs = jax.scipy.linalg.cho_solve((AAT1, True), Ax1)
    # AAT1 = AAT * e1 + a[:, None, :] * a[:, :, None]
    # coefs = jnp.linalg.solve(AAT1, Ax1)
    return (AAT1, Ax1, t, freqs, harms, alpha), coefs


def exp_rolling_lombscargle_cho(times, acceleration, freqs, n_harms, alpha=1e-1):
    n_freqs = len(freqs)
    harms = np.arange(1, n_harms + 1)
    xt = jnp.c_[times, acceleration]
    AAT = jnp.concatenate([
        jnp.eye(2 * n_harms + 1)[None, ...] for i in range(n_freqs)
    ])
    Ax = jnp.zeros((n_freqs, 2 * n_harms + 1))
    init = (AAT, Ax, times[0], freqs, harms, alpha)
    carry, coefs = jax.lax.scan(
        _exp_rolling_lombscargle_cho, init, xt
    )
    return coefs