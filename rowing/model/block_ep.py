"""Block expectation-propagation over the crew-speed model.

Idea: a block factor *is* :class:`~rowing.model.performance.competition_model.PerformanceGP` on a
subset of boats -- the simple GP already correlates two boat-results by time distance whenever they
share athletes. So instead of EP with one-competition factors, we fit **overlapping time windows**
jointly and pass messages on the athletes that span windows.

The shared variable is each athlete's latent score at a reference time ``t_ref`` -- exactly what
``PerformanceGP.predict_athletes_score`` computes -- so a single window covering all boats reduces
to the exact GP (the anchor for verification).

:func:`run_ep` is the damped EP scheduler: it reconciles, across windows, both the cross-window
athletes (score at ``t_ref``) and the *global* boat-class baselines (moved out of each window's
kernel into a shared latent, its cavity folded into the boat covariance like the competition
factor). :func:`block_athlete_scores` is a simpler one-pass product-of-experts baseline.
"""

from typing import NamedTuple

import numpy as np
from scipy import linalg as sla

from rowing.model.performance.competition_model import (
    PerformanceGP,
    get_athlete_kernel,
    load_kernel_haiku_params,
)


def time_window_masks(years, width, step=None):
    """Boolean boat masks for overlapping time windows over ``years``.

    ``width`` is the window span (in years); ``step`` (default = ``width``, i.e. no overlap) is the
    gap between consecutive window starts, so ``overlap = width - step``. Windows tile ``[min, max]``
    and the last is clamped to include the final year, so every boat falls in >= 1 window.
    """
    years = np.asarray(years)
    lo, hi = float(years.min()), float(years.max())
    step = width if step is None else step
    if hi <= lo:
        return [np.ones(len(years), bool)]
    starts = np.arange(lo, hi, step)
    masks = []
    for s in starts:
        e = s + width
        m = (years >= s) & (years <= e if e >= hi else years < e)
        if m.any():
            masks.append(m)
    # guarantee coverage: any boat in no window joins the nearest (last) window
    covered = np.any(masks, axis=0)
    if not covered.all():
        masks[-1] = masks[-1] | ~covered
    return masks


def athlete_windows(mi, masks):
    """For each global athlete code, the indices of the windows it races in.

    Uses the seat COO (``seat_athlete``/``seat_boat``): an athlete is "in" a window if any of its
    seats belongs to a boat in that window. Returns ``{athlete_code: [window indices]}`` and the set
    of athletes that span more than one window (the EP boundary).
    """
    seat_boat = np.asarray(mi.seat_boat)
    seat_ath = np.asarray(mi.seat_athlete)
    windows = {}
    for w, mask in enumerate(masks):
        mask = np.asarray(mask)
        ath_in_w = np.unique(seat_ath[mask[seat_boat]])
        for a in ath_in_w:
            windows.setdefault(int(a), []).append(w)
    shared = {a for a, ws in windows.items() if len(ws) > 1}
    return windows, shared


def window_athlete_message(mi_window, t_ref, *, params=None, **kernels):
    """Per-window message: posterior mean/var of each athlete's score at ``t_ref``.

    Builds ``PerformanceGP`` on the window subset and returns the diagonal Gaussian over all athlete
    codes (mean, var). Athletes with no seats in the window carry the GP prior (near-zero info); the
    caller restricts to the window's actual athletes via :func:`athlete_windows`.
    """
    gp = PerformanceGP.from_inputs(mi_window, params=params, **kernels)
    y_ath, cov_ath = gp.predict_athletes_score(t_ref)
    return np.asarray(y_ath), np.asarray(cov_ath).diagonal()


def fuse_messages(messages, athlete_windows_map, n_athletes, prior_var):
    """Product-of-experts fusion of per-window athlete-score messages (Stage-1 baseline).

    Each window contributes a diagonal Gaussian on its athletes; an athlete's fused posterior is the
    product of its windows' messages divided by the shared prior over-counting (one prior kept).
    Returns (mean, var) per global athlete code. (Full EP with cavities iterates on top of this in
    :func:`run_ep`.)
    """
    # Work in natural params (precision, precision*mean), starting from the shared prior N(0, prior_var).
    # Each window's message = its posterior minus the prior it already includes, so summing messages
    # over an athlete's windows + the single prior fuses the evidence (one window -> exact posterior).
    prec = np.full(n_athletes, 1.0 / prior_var)
    pm = np.zeros(n_athletes)
    for w, (mean, var) in enumerate(messages):
        ath = np.array([a for a, ws in athlete_windows_map.items() if w in ws], dtype=int)
        if ath.size == 0:
            continue
        prec[ath] += 1.0 / var[ath] - 1.0 / prior_var  # posterior precision - prior precision
        pm[ath] += mean[ath] / var[ath]  # posterior precision-mean - prior precision-mean (=0)
    var = 1.0 / prec
    return pm * var, var


def athlete_prior_var(t_ref, *, params=None, athlete_kernel=get_athlete_kernel, **_):
    """Prior variance of an athlete's score at ``t_ref`` (the athlete kernel's marginal at t_ref)."""
    ak = athlete_kernel()
    if params is not None:
        load_kernel_haiku_params(ak, params)
    return float(np.asarray(ak.K(np.r_[t_ref], np.r_[t_ref])).reshape(()))


def block_athlete_scores(mi, masks, t_ref, *, params=None, **kernels):
    """Fuse per-window athlete-score messages into a posterior over all athletes at ``t_ref``.

    Returns (mean, var) per global athlete code and the set of cross-window (shared) athletes. With a
    single window covering all boats this equals ``PerformanceGP.predict_athletes_score(t_ref)``.
    """
    aw, shared = athlete_windows(mi, masks)
    messages = [window_athlete_message(mi.subset(np.asarray(m)), t_ref, params=params, **kernels) for m in masks]
    prior_var = athlete_prior_var(t_ref, params=params, **kernels)
    mean, var = fuse_messages(messages, aw, mi.n_athletes, prior_var)
    return mean, var, shared


class EPResult(NamedTuple):
    athlete_mean: np.ndarray  # per global athlete code, score at t_ref
    athlete_var: np.ndarray
    class_mean: np.ndarray  # per boat-class code, global baseline
    class_var: np.ndarray
    shared: set  # cross-window athletes
    history: list  # max |Delta posterior mean| per sweep


def window_factor(gp, mi_window, t_ref, cond_ath, cav_ath_mean, cav_ath_var, cav_cls_mean, cav_cls_var, jitter=1e-9):
    """Posterior over a window's athlete scores (at t_ref) and boat-class baselines.

    Boat-class is a *global* latent removed from the window kernel: its cavity is folded into the
    boat covariance (``K_eff = K_block + W_bc diag(cav) W_bc^T``, ``y_eff = y - W_bc cav_mean``),
    matching the competition-factor pattern, so one window reduces to the exact GP. Cross-window
    athletes (``cond_ath``) are conditioned via score pseudo-observations on top.
    """
    K_ath, K_race, K_bc = gp.get_kernels()
    noise = float(np.exp(np.asarray(gp.log_noise[...])))
    y = np.asarray(gp.y)
    nb = len(y)
    K_block = np.asarray(K_ath) + np.asarray(K_race) + noise * np.eye(nb)

    W_ath = np.asarray(gp.W_athlete)
    na = W_ath.shape[1]
    k_pred = np.asarray(gp.athlete_kernel.K(np.r_[t_ref], gp.years))[0]
    k00 = float(np.asarray(gp.athlete_kernel.K(np.r_[t_ref], np.r_[t_ref])).reshape(()))
    C_ath = k_pred[:, None] * W_ath  # cross-cov boat <-> athlete score

    W_bc = np.asarray(mi_window.one_hot("class"))  # (nb, n_classes), global codes
    nc = W_bc.shape[1]
    cav_cls_mean = np.asarray(cav_cls_mean)
    cav_cls_var = np.asarray(cav_cls_var)

    # fold the boat-class cavity into the boat covariance / mean
    K_eff = K_block + (W_bc * cav_cls_var) @ W_bc.T
    y_eff = y - W_bc @ cav_cls_mean

    cond_ath = np.asarray(cond_ath, int)
    if cond_ath.size:
        Cs = C_ath[:, cond_ath]
        Kaug = np.block([[K_eff, Cs], [Cs.T, k00 * np.eye(cond_ath.size) + np.diag(np.asarray(cav_ath_var))]])
        Kaug[np.diag_indices_from(Kaug)] += jitter
        L = np.linalg.cholesky(Kaug)
        a = sla.cho_solve((L, True), np.concatenate([y_eff, np.asarray(cav_ath_mean)]))
        m = cond_ath.size
    else:
        L = np.linalg.cholesky(K_eff + jitter * np.eye(nb))
        a = sla.cho_solve((L, True), y_eff)
        m = 0

    # athlete posteriors (all athletes)
    cross_ath = np.zeros((na, nb + m))
    cross_ath[:, :nb] = C_ath.T
    if m:
        cross_ath[cond_ath, nb + np.arange(m)] = k00
    ath_mean = cross_ath @ a
    Lx = sla.solve_triangular(L, cross_ath.T, lower=True)
    ath_var = k00 - np.square(Lx).sum(0)

    # boat-class posteriors: prior mean + cov W_bc^T (K_eff-solve), cov diag minus the data reduction
    cross_bc = np.zeros((nc, nb + m))
    cross_bc[:, :nb] = cav_cls_var[:, None] * W_bc.T
    cls_mean = cav_cls_mean + cross_bc @ a
    Lb = sla.solve_triangular(L, cross_bc.T, lower=True)
    cls_var = cav_cls_var - np.square(Lb).sum(0)
    return ath_mean, ath_var, cls_mean, cls_var


def run_ep(mi, masks, t_ref, *, n_iter=20, damping=0.5, params=None, **kernels):
    """Damped expectation propagation over overlapping time-window blocks.

    Reconciles, across windows, both the cross-window athletes (score at ``t_ref``) and the
    *global* boat-class baselines. Returns an :class:`EPResult`. One window covering all boats
    reproduces the exact ``PerformanceGP`` posteriors (athletes and boat-class).
    """
    mi_w = [mi.subset(np.asarray(m)) for m in masks]
    gps = [PerformanceGP.from_inputs(w, params=params, **kernels) for w in mi_w]
    aw, shared = athlete_windows(mi, masks)
    window_ath = [np.array(sorted(a for a, ws in aw.items() if w in ws), int) for w in range(len(masks))]
    window_cls = [np.unique(np.asarray(w.boat_class)) for w in mi_w]
    na, nc = mi.n_athletes, mi.n_classes
    k00 = athlete_prior_var(t_ref, params=params, **kernels)
    boatclass_var = float(np.asarray(gps[0].boatclass_var.value).reshape(()))

    # combined latent space: [0:na) athlete scores, [na:na+nc) boat-class baselines
    prior_var = np.concatenate([np.full(na, k00), np.full(nc, boatclass_var)])
    msg_prec = [np.zeros(na + nc) for _ in masks]
    msg_pm = [np.zeros(na + nc) for _ in masks]

    def posterior():
        prec = 1.0 / prior_var + sum(msg_prec)
        pm = sum(msg_pm)
        return pm / prec, 1.0 / prec

    history = []
    prev, _ = posterior()
    for _ in range(n_iter):
        for w, gp in enumerate(gps):
            ath, cls = window_ath[w], window_cls[w]
            idx = np.concatenate([ath, na + cls])
            if idx.size == 0:
                continue
            qm, qv = posterior()
            cav_prec = np.maximum(1.0 / qv[idx] - msg_prec[w][idx], 1e-8)
            cav_pm = qm[idx] / qv[idx] - msg_pm[w][idx]
            cav_var = 1.0 / cav_prec
            cav_mean = cav_pm * cav_var
            ca_m, ca_v = cav_mean[: ath.size], cav_var[: ath.size]  # athlete cavities
            cc_m, cc_v = cav_mean[ath.size :], cav_var[ath.size :]  # class cavities (this window)

            # full boat-class cavity vectors over all nc (non-window classes -> harmless prior)
            cls_m_full = np.zeros(nc)
            cls_v_full = np.full(nc, boatclass_var)
            cls_m_full[cls] = cc_m
            cls_v_full[cls] = cc_v

            sh = np.isin(ath, list(shared))  # only cross-window athletes are conditioned
            am, av, cm, cv = window_factor(gp, mi_w[w], t_ref, ath[sh], ca_m[sh], ca_v[sh], cls_m_full, cls_v_full)

            post_prec = np.concatenate([1.0 / av[ath], 1.0 / cv[cls]])
            post_pm = np.concatenate([am[ath] / av[ath], cm[cls] / cv[cls]])
            msg_prec[w][idx] = (1 - damping) * msg_prec[w][idx] + damping * (post_prec - cav_prec)
            msg_pm[w][idx] = (1 - damping) * msg_pm[w][idx] + damping * (post_pm - cav_pm)

        mean, _ = posterior()
        history.append(float(np.nanmax(np.abs(mean - prev))))
        prev = mean

    mean, var = posterior()
    return EPResult(mean[:na], var[:na], mean[na:], var[na:], shared, history)
