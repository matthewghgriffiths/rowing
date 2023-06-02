
from typing import NamedTuple

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp

from .. import utils

T_BASE = 4 * 60

def to_year(dt):
    return dt.year + dt.dayofyear / 365.25

def power_to_pace(power):
    return (2.8 / power)**(1/3)

def pace_to_power(pace):
    return 2.8 * pace ** -3 

def estimate_distance_time(D, P, coef, t_base=T_BASE):
    return ((2.8 * D**3)/P/t_base**coef)**(1/(3 - coef))

def estimate_distance_time_jac(D, P, coef, t_base=T_BASE):
    t, vjp = jax.vjp(
        lambda P, coef: estimate_distance_time(D, P, coef, t_base=t_base),
        P, coef
    )
    return t, vjp(jnp.ones_like(t))

def predict_distance_split(distance, P, P_std, coef, coef_std, t_base=T_BASE):
    est_time, (jac_P, jac_coef) = estimate_distance_time_jac(
        distance, P, coef, t_base=t_base)
    est_time_std = np.sqrt((jac_P * P_std)**2 + (coef_std * jac_coef)**2)
    est_split = est_time * 500 / distance
    est_split_std = est_time_std * 500 / distance
    return est_split, est_split_std

def estimate_split(t, P, coef, t_base=T_BASE):
    return 500 * power_to_pace(P * (t / t_base)**(-coef))

def estimate_split_jac(t, P, coef, t_base=T_BASE):
    split, vjp = jax.vjp(
        lambda P, coef: estimate_split(t, P, coef, t_base=t_base),
        P, coef
    )
    return split, vjp(jnp.ones_like(split))

def predict_time_split(t, P, P_std, coef, coef_std, t_base=T_BASE):
    est_split, (jac_P, jac_coef) = estimate_split_jac(t, P, coef, t_base=t_base)
    est_split_std = np.sqrt((jac_P * P_std)**2 + (coef_std * jac_coef)**2)
    return est_split, est_split_std

### Power law modelling

def create_power_law_data(year, duration, power, t_base=T_BASE, **kwargs):
    kwargs.update({
        "t": year,
        "log_t": np.log(duration / t_base),
        "log_p": np.log(power)
    })
    return pd.DataFrame(kwargs).sort_values("t")

class PowerLawInput(NamedTuple):
    t: np.ndarray
    W: np.ndarray
    y: np.ndarray

def create_power_law_inputs(power_data):
    data = power_data.sort_values("t")
    t = jnp.array(data.t)
    W = jnp.c_[
        jnp.ones(len(data.log_t)),
        jnp.array(- data.log_t), 
    ]
    y = jnp.array(data.log_p)
    return PowerLawInput(t, W, y)

def power_posterior(Z, Z_std):
    logp, coef = Z.T
    logp_std, coef_std = Z_std.T
    P = jnp.exp(logp)
    P_std = P * logp_std
    return P, P_std, coef, coef_std

### Plotting

def plot_posterior(t, mean, std, ax=None, alpha=0.3, format_splits=True, **kwargs):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    m = np.reshape(mean, (len(mean), -1))
    s = np.reshape(std, (len(mean), -1))
    lines = []
    bands = []
    for m_i, s_i in zip(m.T, s.T):
        line, = ax.plot(t, m_i, **kwargs)
        lines.append(line)
        bands.append(ax.fill_between(
            t, m_i - s_i, m_i + s_i,
            facecolor=line.get_color(), 
            alpha=alpha
        ))

    format_splits and utils.format_yaxis_splits(ax)
    return lines, bands


def plot_athlete_data(data, ax=None, ls=':', marker='+', label='', format_splits=False, **kwargs):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    lines = []
    for piece, piece_data in data.groupby("piece"):
        l, = ax.plot(
            piece_data.year, 
            piece_data.split.dt.total_seconds(), 
            label=label + piece, 
            ls=ls, marker=marker, 
            **kwargs
        )
        lines.append(l)

    format_splits and utils.format_yaxis_splits(ax)
    return lines


def make_predictions(t, make_model, params, inputs, t_pred=None, t_base=T_BASE, W=None):
    from .gp import transform
    model = make_model(params, *inputs)
    Z, Z_var = transform(model.predict_coef_var).apply(params, t)
    Z_std = Z_var**0.5
    P, P_std, coef, coef_std = power_posterior(Z, Z_std)
    t_pred = t_pred or t_base
    split, split_std = predict_time_split(
        t_pred, P, P_std, coef, coef_std, t_base=t_base
    )
    preds = {
        "P": P, "P_std": P_std, 
        "coef": coef, "coef_std": coef_std,
        "split": split, "split_std": split_std,
    }
    if W is not None:
        preds['pred_log_p'], preds['log_p_var'] = transform(model.predict_var).apply(params, t, W)

    for i, (z, z_std) in enumerate(zip(Z.T, Z_std.T)):
        preds[f"z_{i}"] = z
        preds[f"z_std_{i}"] = z_std
    return preds


def plot_model_posterior(
        t, make_model, params, inputs, t_base=T_BASE, t_plot=None, axes=None, figsize=(10, 8), **kwargs
):
    import matplotlib.pyplot as plt

    preds = make_predictions(t, make_model, params, inputs, t_base=t_base)
    P, P_std, coef, coef_std = (
        preds[k] for k in ("P", "P_std", "coef", "coef_std")
    )

    if axes is None:
        f, axes = plt.subplots(2, sharex=True, figsize=figsize)
    else:
        f = plt.gcf()

    ax = axes[0]
    plot_posterior(
        t, coef, coef_std, format_splits=False, ax=ax, **kwargs
    )
    ax.set_ylabel("Power law: predicted exponent")

    ax = axes[1]
    if t_plot:
        tplot_min = t_plot/60
        
        if tplot_min < 1:
            label=f"Power law: predicted {t_plot} s split"
        else:
            if tplot_min % 1 == 0:
                tplot_min = int(tplot_min)
            else:
                tplot_min = round(tplot_min, 2)

            label=f"Power law: predicted {tplot_min} min split"
        
        split, split_std = predict_time_split(
            t_plot, P, P_std, coef, coef_std, t_base=t_base)

        plot_posterior(
            t, split, split_std, format_splits=False, ax=ax, 
            label=label,
            **kwargs
        )

    return f, axes, preds


def plot_model_preds(
        t, make_model, params, inputs, 
        distances=(2000, 5000), times=(180, 300, 720), t_base=T_BASE, 
        alpha=0.3, figsize=(10, 8), axes=None, **kwargs
):
    import matplotlib.pyplot as plt

    if axes is None:
        f, axes = plt.subplots(2, sharex=True, figsize=figsize)
    else:
        f = plt.gcf()

    _, _, preds = plot_model_posterior(
        t, make_model, params, inputs, axes=axes, alpha=alpha, t_base=t_base, **kwargs)
    P, P_std, coef, coef_std = (
        preds[k] for k in ("P", "P_std", "coef", "coef_std")
    )

    ax = axes[-1]
    for D in distances:
        Dkm = D/1000
        if Dkm % 1 == 0:
            Dkm = int(Dkm)
            
        plot_posterior(
            t, *predict_distance_split(
                D, P, P_std, coef, coef_std, t_base=t_base, 
            ),
            label=f"Power law: predicted {Dkm}k split",
            format_splits=False, alpha=alpha, ax=ax    
        )
    for duration in times:
        dmin = duration/60
        if dmin % 1 == 0:
            dmin = int(dmin)

        plot_posterior(
            t, *predict_time_split(
                duration, P, P_std, coef, coef_std, t_base=t_base, 
            ),
            label=f"Power law: predicted {dmin} min split",
            format_splits=False, alpha=alpha, ax=ax    
        )

    utils.format_yaxis_splits(ax)
    f.tight_layout()

    return f, axes, {}