
import pandas as pd 
import jax 
import jax.numpy as jnp

def zero_crossings(carry, x):
    data, params = carry
    t0, a0, v0, s0, sv0, var_s0 = data 
    lam, = params

    t1, a1 = x
    dt = t1 - t0 
    v1 = v0 + dt * (a1 + a0) / 2

    alpha = jnp.exp(- lam * dt)
    d1 = a1 - s0 
    s1 = s0 + d1 * alpha
    sv1 = sv0 + dt * (s1 + s0)/2
    var_s1 = (1 - alpha) * (var_s0 + alpha * d1**2)

    sgn_chg = jnp.sign(s0) != jnp.sign(s1)
    next_carry = (
        jnp.r_[t1, a1, v1, s1, sv1, var_s1], 
        (lam,)
    )
    val = jnp.r_[a1, v1, s1, sv1, var_s1, sgn_chg]

    return next_carry, val


def find_smoothed_crossings(times, acceleration, lam=100.):
    tx = jnp.c_[times, acceleration]

    init = (jnp.zeros(6), (lam,))
    carry, vals = jax.lax.scan(
        zero_crossings, init, tx
    )
    return pd.DataFrame(
        vals, 
        index=times, 
        columns=['a', 'v', 'smooth', 'smooth_v', 'var_smooth_a', 'sgn_chg']
    )

def find_strokes(crossings, decel_thresh=0.5, accel_thresh=0.5, search_back=1.):
    peak_data = crossings.loc[crossings.sgn_chg == 1].copy()
    peak_data['DeltaV'] = crossings.v.loc[peak_data.index].diff()

    strokes = peak_data.loc[
        (peak_data.DeltaV < - decel_thresh)
        & (peak_data.DeltaV.shift(-1) > accel_thresh)
    ].copy()

    strokes['stroke_end'] = strokes.apply(
        lambda x: crossings.v[
            x.name - 1 : x.name
        ].idxmin(), 
        axis=1
    )
    strokes['stroke_start'] = strokes.stroke_end.shift(fill_value=0)
    strokes['stroke_time'] = strokes.stroke_end - strokes.stroke_start
    strokes['stroke_rate'] = 60 / strokes.stroke_time


    return strokes 