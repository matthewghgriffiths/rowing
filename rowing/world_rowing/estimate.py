

from functools import partial

import pandas as pd

import jax
from jax import numpy as jnp, lax

from . import livetracker


diff0 = partial(jnp.diff, prepend=0)
vdiag = jax.vmap(jnp.diag)
vmatmul = jax.vmap(jnp.matmul)
vdiagonal = jax.vmap(jnp.diagonal)


@jax.jit
def tridiagonal_matmul(dl, d, du, b):
    return (
        b * d
        + jnp.roll(b * jnp.roll(dl.at[..., 0].set(0), -1, axis=-1), 1, axis=-1)
        + jnp.roll(b * jnp.roll(du.at[..., -
                   1].set(0), 1, axis=-1), -1, axis=-1)
    )


def ffill(x, init=None, **kwargs):
    def something(x0, x):
        x1 = lax.select(jnp.isnan(x), x0, x)
        return x1, x1
    if init is None:
        if kwargs.get("reverse"):
            init = x[-1]
        else:
            init = x[0]

    return lax.scan(
        something, init, x, **kwargs
    )[1]


bfill = partial(ffill, reverse=True)


@jax.jit
def interpna(x, f, x0=None, f0=None):
    if f0 is None or x0 is None:
        i = (jnp.isfinite(f) & jnp.isfinite(x)).argmax()
        f0 = f[i]
        x0 = x[i]

    def _ffill(carry, xs):
        x0, f0 = carry
        x, f = xs
        valid = jnp.isfinite(f)
        x1 = lax.select(valid, x, x0)
        f1 = lax.select(valid, f, f0)
        return ((x1, f1),)*2

    _, (xp, fp) = lax.scan(_ffill, (x0, f0), (x, f))
    return jnp.interp(x, xp, fp)


def _step_dist(carry, xs):
    x0, x0_prec, p0, p0_prec = carry
    x1, x1_prec, p1, p1_prec = xs

    dxi = x1 - x0
    dxi_prec = x1_prec * x0_prec / (x1_prec + x0_prec)
    pi = jnp.where(jnp.isfinite(p1), p1, p0)
    pi_prec = jnp.where(p1_prec > 0, p1_prec, p0_prec)

    pdxi = pi * dxi
    pdxi_prec = pi_prec * dxi_prec / (pi * pi_prec + dxi * dxi_prec + 1)
    dti_prec = jnp.nansum(pdxi_prec)
    dti = jnp.nansum(pdxi * pdxi_prec) / dti_prec
    dti_var = (1 + jnp.nansum(pdxi_prec * jnp.square(pdxi - dti))) / dti_prec

    dxi_update = ~ jnp.isfinite(dxi)
    dxi = jnp.where(dxi_update, dti / pi, dxi)
    dxi_prec = jnp.where(
        dxi_update,
        pi_prec / (
            pi_prec * dti_var / pi**2
            + dti**2 / pi**4 + dti_var / pi**4),
        dxi_prec
    )
    xi_update = ~ jnp.isfinite(x1)

    xi = jnp.where(xi_update, x0 + dxi, x1)
    xi_prec = jnp.where(
        xi_update, x0_prec * dxi_prec / (x0_prec + dxi_prec), x1_prec)
    out = (xi, xi_prec, pi, pi_prec, dxi, 1 / dxi_prec)
    return out[:4], out


@jax.jit
def estimate_times(
        inter_times, inter_dists,
        dist, dist_var,
        pace, pace_var,
        inter_time_var=1e-4/12,
        inter_dist_var=1e-4/12,
        t_prior_prec=0,
        race_distance=2000,
):
    # Find points to insert at
    it = inter_times.T
    boats = jnp.arange(it.shape[1])
    id, iboat = jnp.meshgrid(jnp.arange(
        inter_dists.size), boats, indexing='ij')

    itf0, idf0, ibf0 = it.flatten(), id.flatten(), iboat.flatten()

    isort = itf0.argsort()
    itf, idf, ibf = itf0[isort], idf0[isort], ibf0[isort]

    insertat = jax.vmap(jnp.searchsorted, (1, None))(
        dist, inter_dists)[ibf, idf]
    insertedat = insertat + jnp.arange(insertat.size)
    points = jnp.ones(
        len(dist) + insertedat.size, bool).at[insertedat].set(False)

    inter_x = jnp.where(ibf[:, None] == boats[None, :],
                        inter_dists[idf][:, None], jnp.nan)
    inter_x_prec = jnp.where(
        ibf[:, None] == boats[None, :], 1/inter_dist_var, 0)

    x0 = jax.vmap(jnp.insert, (1, None, 1))(dist, insertat, inter_x).T
    x0_prec = jax.vmap(jnp.insert, (1, None, 1))(
        1 / dist_var, insertat, inter_x_prec).T

    p = ffill(jax.vmap(jnp.insert, (1, None, None))(pace, insertat, jnp.nan).T)
    p_var = ffill(jax.vmap(jnp.insert, (1, None, None))
                  (pace_var, insertat, jnp.nan).T)
    p_prec = jnp.where((x0 < race_distance) & jnp.isfinite(p), 1/p_var, 0)

    carry = (
        jnp.zeros_like(x0[0], dtype=float),
        jnp.full_like(x0[0], 1/inter_dist_var, dtype=float),
        pace[0],
        1 / pace_var[0]
    )
    _, (x, x_prec, p, p_prec, dx, dx_var) = jax.lax.scan(
        _step_dist, carry, (x0, x0_prec, p, p_prec))
    x_var = 1 / x_prec
    p_var = 1 / p_prec

    pdx = p * dx
    pdx_var = dx**2 * p_var + p**2 * dx_var + p_var * dx_var
    pdx_prec = 1 / pdx_var
    pdx_prec = x_prec * p_prec / (dx**2 * x_prec + p**2 * p_prec + 1)

    dt_var = 1 / jnp.nansum(pdx_prec, axis=1)
    dt = dt_var * jnp.nansum(pdx * pdx_prec, axis=1)
    dt_var += jnp.nansum(
        pdx_prec * jnp.square(pdx - dt[:, None]), axis=1
    ) * dt_var

    bd = jnp.ones(dt.size)
    bl, b0 = - bd.at[0].set(0), jnp.zeros_like(bd)

    bbd2 = bd**2 / dt_var
    bbd = (bbd2 + bl**2 / jnp.roll(dt_var, 1, axis=-1))
    bbdu = - bbd2.at[-1].set(0)
    bbdl = jnp.roll(bbdu, 1, axis=-1)

    est_t = jax.lax.linalg.tridiagonal_solve(bl, bd, b0, dt[:, None])[:, 0]
    t = est_t.at[insertedat].set(itf)
    t_prec = jnp.full_like(
        t, t_prior_prec).at[insertedat].set(1 / inter_time_var)
    prec_t = (t * t_prec + tridiagonal_matmul(bbdl, bbd, bbdu, est_t))
    prec_d = bbd + t_prec
    post_t = jax.lax.linalg.tridiagonal_solve(
        bbdl, prec_d, bbdu, prec_t[..., None])[..., 0]
    post_t_covar = jax.lax.linalg.tridiagonal_solve(
        bbdl, prec_d, bbdu,
        jax.lax.linalg.tridiagonal_solve(
            bbdl, prec_d, bbdu, jnp.eye(len(bbd))).T
    )
    post_t = jax.lax.cummax(post_t)

    post_dt = tridiagonal_matmul(bl, bd, b0, post_t)
    post_dt_covar = tridiagonal_matmul(
        bl, bd, b0, tridiagonal_matmul(bl, bd, b0, post_t_covar).T)

    post_x = jax.vmap(interpna, (None, 1))(post_t, x).T
    post_x_var = jax.vmap(interpna, (None, 1))(post_t, x_var).T

    post_pace = jax.vmap(interpna, (None, 1))(post_t, p).T
    post_pace_var = jax.vmap(interpna, (None, 1))(post_t, p_var).T

    win_time = jnp.nanmin(inter_times[:, -1])
    win_speed = inter_dists[-1] / win_time
    return (
        (post_t, post_t_covar.diagonal()),
        (post_dt, post_dt_covar.diagonal()),
        (post_x, post_x_var), (dx, dx_var), (post_pace, post_pace_var),
        (win_speed, race_distance, post_t_covar, points)
    )


def parse_livetracker_data(data):
    live_boat_data = livetracker.parse_livetracker_data(data)
    intermediates = livetracker.parse_intermediates_data(data)
    race_distance = data['config']['plot']['totalLength']

    live_distance = live_boat_data.distanceTravelled
    live_speed = live_boat_data.metrePerSecond.replace(0, jnp.nan)
    live_speed = live_speed[
        True
        & live_distance.notna().any(axis=1)
        & live_speed.notna().any(axis=1)
        & (live_distance != 0).all(axis=1)
        & ~ live_distance.apply(lambda s: s.duplicated()).all(1)
    ]
    live_distance = live_distance.loc[live_speed.index, live_speed.columns]
    live_pace = 1 / live_speed

    inters = intermediates[
        ['distance', 'ResultTime']
    ].stack().swaplevel()
    inters['ResultTime'] = inters.ResultTime.dt.total_seconds()
    inters['distance'] = inters.distance.astype(int)
    inters = inters.reset_index().set_index(
        ["distance", 'boat']
    ).ResultTime.unstack()[live_pace.columns]
    inters.loc[0] = 0
    inters = inters.sort_index()[live_pace.columns]

    speed = live_speed.ffill(axis=0).values
    speed_var = jnp.where(
        live_pace.notna().values, 1e-2 / 12, 100).at[0].set(10)

    pace = 1 / speed
    pace_var = speed_var / speed**4

    dist = live_distance.values
    dist_var = jnp.where(
        live_distance.notna().values & (dist < race_distance), 1/12, 1e2
    ).at[dist == race_distance].set(1)

    inter_times = inters.values.T
    inter_dists = inters.index.values
    return {
        'inter_times': inter_times,
        'inter_dists': inter_dists,
        'dist': dist,
        'dist_var': dist_var,
        'pace': pace,
        'pace_var': pace_var,
        'race_distance': race_distance
    }


def parse_livetracker_times(
        data,
        inter_time_var=1e-4/12,
        inter_dist_var=1e-4/12,
        t_prior_prec=0
):
    live_boat_data = livetracker.parse_livetracker_data(data)
    intermediates = livetracker.parse_intermediates_data(data)
    race_distance = data['config']['plot']['totalLength']

    # .replace(race_distance, np.nan)
    live_distance = live_boat_data.distanceTravelled
    live_speed = live_boat_data.metrePerSecond.replace(0, jnp.nan)

    live_speed = live_speed[
        True
        & live_distance.notna().any(axis=1)
        & live_speed.notna().any(axis=1)
        & (live_distance != 0).all(axis=1)
        & ~ live_distance.apply(lambda s: s.duplicated()).all(1)
    ]
    live_distance = live_distance.loc[live_speed.index, live_speed.columns]
    live_pace = 1 / live_speed

    inters = intermediates[
        ['distance', 'ResultTime']
    ].stack().swaplevel()
    inters['ResultTime'] = inters.ResultTime.dt.total_seconds()
    inters['distance'] = inters.distance.astype(int)
    inters = inters.reset_index().set_index(
        ["distance", 'boat']
    ).ResultTime.unstack()[live_pace.columns]
    inters.loc[0] = 0
    inters = inters.sort_index()[live_pace.columns]

    dist = live_distance.values
    dist_var = jnp.where(
        live_distance.notna().values & (dist < race_distance), 1/12, 1e2
    ).at[dist == race_distance].set(1)

    delta = diff0(dist, axis=0)
    speed = live_speed.ffill(axis=0).values

    speed_var = speed**2 / delta / 12
    speed_var = jnp.where(
        jnp.isfinite(speed_var), speed_var, 100).at[0].set(10)

    pace = 1 / speed
    pace_var = speed_var / speed**4

    inter_times = inters.values.T
    inter_dists = inters.index.values

    return estimate_times(
        inter_times, inter_dists,
        dist, dist_var,
        pace, pace_var,
        inter_time_var=1e-4/12,
        inter_dist_var=1e-4/12,
        t_prior_prec=0,
        race_distance=race_distance
    )
