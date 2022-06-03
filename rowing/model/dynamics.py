
import numpy as np

import jax 
import jax.numpy as jnp

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
    j = jnp.arange(1, cs.size + 1) * jnp.pi * 2

    cj, dj = cs/j, ds/j
    cjj, djj = cj/j, dj/j
    
    ns = n * j 
    n1 = n + s * dt
    n1s = n1 * j 
    
    cosn = jnp.cos(ns)
    sinn = jnp.sin(ns)
    cosn1 = jnp.cos(n1s)
    sinn1 = jnp.sin(n1s)

    a1 = (cs * cosn + ds * sinn).sum()
    v1 = v0 + (cj * sinn - cj * sinn).sum() / s
    dx = v0 * dt + (
        cjj * cosn + djj * sinn
        - cjj * cosn1 - djj * sinn1
    ).sum() / s /s

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