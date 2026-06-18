"""Block expectation-propagation over the crew-speed model.

Idea: a block factor *is* :class:`~rowing.model.performance.competition_model.PerformanceGP` on a
subset of boats -- the simple GP already correlates two boat-results by time distance whenever they
share athletes. So instead of EP with one-competition factors, we fit **overlapping time windows**
jointly and pass messages on the athletes that span windows.

The shared variable is each athlete's latent score at a reference time ``t_ref`` -- exactly what
``PerformanceGP.predict_athletes_score`` computes -- so a single window covering all boats reduces
to the exact GP (the anchor for verification).

Stage 1 here: the time-window partition, the per-window athlete-score *message*, and a
product-of-experts fusion of windows. The damped EP scheduler (re-conditioning each window on the
cavity, iterating) builds on this in :func:`run_ep`.
"""

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


def window_conditioned_scores(gp, t_ref, shared_codes, cav_mean, cav_var, jitter=1e-9):
    """Posterior mean/var of every athlete's ``t_ref`` score for this window's GP, optionally
    conditioned on a cavity Gaussian over ``shared_codes`` (cross-window athletes).

    The cavity is incorporated as pseudo-observations of the athlete-score functionals: we augment
    the boat GP with extra "observations" ``s_a = cav_mean_a (+/- cav_var_a)`` whose cross-covariance
    to boat ``i`` is ``k(t_ref, year_i) * W[i, a]`` and whose prior (co)variance is ``k00`` (the
    athlete kernel at t_ref). With ``shared_codes`` empty this reduces to predict_athletes_score.
    """
    K = np.asarray(gp.get_jitter_kernel())
    y = np.asarray(gp.y)
    W = np.asarray(gp.W_athlete)
    k_pred = np.asarray(gp.athlete_kernel.K(np.r_[t_ref], gp.years))[0]
    k00 = float(np.asarray(gp.athlete_kernel.K(np.r_[t_ref], np.r_[t_ref])).reshape(()))
    C = k_pred[:, None] * W  # (n_boats, n_athletes): cross-cov boat <-> athlete score
    nb, na = W.shape
    shared_codes = np.asarray(shared_codes, dtype=int)

    if shared_codes.size:
        Cs = C[:, shared_codes]  # (nb, |S|)
        Kaug = np.block([[K, Cs], [Cs.T, k00 * np.eye(shared_codes.size) + np.diag(np.asarray(cav_var))]])
        Kaug[np.diag_indices_from(Kaug)] += jitter
        yaug = np.concatenate([y, np.asarray(cav_mean)])
        L = np.linalg.cholesky(Kaug)
        a = sla.cho_solve((L, True), yaug)
        cross = np.zeros((na, nb + shared_codes.size))
        cross[:, :nb] = C.T
        cross[shared_codes, nb + np.arange(shared_codes.size)] = k00  # athlete's score <-> its own pseudo-obs
    else:
        Kaug = K + jitter * np.eye(nb)
        L = np.linalg.cholesky(Kaug)
        a = sla.cho_solve((L, True), y)
        cross = C.T

    mean = cross @ a
    Lc = sla.solve_triangular(L, cross.T, lower=True)
    var = k00 - np.square(Lc).sum(0)
    return mean, var


def run_ep(mi, masks, t_ref, *, n_iter=15, damping=0.5, params=None, **kernels):
    """Damped expectation propagation over time-window blocks.

    Sweeps the windows, re-conditioning each on the cavity (the other windows' messages) about its
    cross-window athletes, until the per-athlete posterior stops moving. Returns (mean, var) per
    global athlete code, the shared-athlete set, and the convergence history (max |Δmean| per sweep).
    """
    gps = [PerformanceGP.from_inputs(mi.subset(np.asarray(m)), params=params, **kernels) for m in masks]
    aw, shared = athlete_windows(mi, masks)
    window_ath = [np.array(sorted(a for a, ws in aw.items() if w in ws), dtype=int) for w in range(len(masks))]
    prior_var = athlete_prior_var(t_ref, params=params, **kernels)
    na = mi.n_athletes

    # site messages per window, in natural params (precision, precision*mean), zero outside its athletes
    msg_prec = [np.zeros(na) for _ in masks]
    msg_pm = [np.zeros(na) for _ in masks]

    def posterior():
        prec = 1.0 / prior_var + sum(msg_prec)
        pm = sum(msg_pm)
        return pm / prec, 1.0 / prec

    history = []
    prev_mean, _ = posterior()
    for _ in range(n_iter):
        for w, gp in enumerate(gps):
            ath = window_ath[w]
            if ath.size == 0:
                continue
            qm, qv = posterior()
            # cavity = posterior / this window's message (natural subtract), guarded positive
            cav_prec = np.maximum(1.0 / qv[ath] - msg_prec[w][ath], 1e-8)
            cav_pm = qm[ath] / qv[ath] - msg_pm[w][ath]
            cav_var = 1.0 / cav_prec
            cav_mean = cav_pm * cav_var

            sh = np.isin(ath, list(shared))
            mean_all, var_all = window_conditioned_scores(gp, t_ref, ath[sh], cav_mean[sh], cav_var[sh])

            # new site message = window posterior / cavity, damped in natural params
            new_prec = 1.0 / var_all[ath] - cav_prec
            new_pm = mean_all[ath] / var_all[ath] - cav_pm
            msg_prec[w][ath] = (1 - damping) * msg_prec[w][ath] + damping * new_prec
            msg_pm[w][ath] = (1 - damping) * msg_pm[w][ath] + damping * new_pm

        mean, _ = posterior()
        history.append(float(np.nanmax(np.abs(mean - prev_mean))))
        prev_mean = mean

    mean, var = posterior()
    return mean, var, shared, history
