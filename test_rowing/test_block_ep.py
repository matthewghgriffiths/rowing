"""Tests for block expectation-propagation (rowing.model.block_ep).

Stage 1: the time-window partition, the per-window athlete-score message, and product-of-experts
fusion. Anchor: a single window covering all boats reproduces the exact
PerformanceGP.predict_athletes_score.
"""

import numpy as np
import pytest

pytest.importorskip("flax")
pytest.importorskip("haiku")  # competition_model still imports haiku until Stage 4

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from rowing.model import block_ep  # noqa: E402
from rowing.model.data import ModelInputs  # noqa: E402
from rowing.model.performance.competition_model import PerformanceGP  # noqa: E402


def _synthetic_mi():
    """8 boats over 4 years (one 2-boat competition per year); 4 athletes recurring across years."""
    years = np.array([2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023], float)
    comp = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    crews = {0: [0, 1], 1: [2, 3], 2: [0, 2], 3: [1, 3], 4: [0, 1], 5: [2, 3], 6: [0, 3], 7: [1, 2]}
    seat_boat = np.array([b for b, ath in crews.items() for _ in ath])
    seat_ath = np.array([a for ath in crews.values() for a in ath])
    rng = np.random.default_rng(0)
    f64, i32 = jnp.float64, jnp.int32
    return ModelInputs(
        y=jnp.asarray(0.9 + 0.02 * rng.standard_normal(8), f64),
        year=jnp.asarray(years, f64),
        hour=jnp.asarray(np.zeros(8), f64),
        boat_venue=jnp.asarray(comp % 2, i32),
        boat_class=jnp.asarray(comp % 2, i32),  # two boat classes, present across all windows
        boat_type=jnp.asarray(comp % 2, i32),
        boat_lane=jnp.asarray(rng.standard_normal(8), f64),
        boat_comp=jnp.asarray(comp, i32),
        seat_boat=jnp.asarray(seat_boat, i32),
        seat_athlete=jnp.asarray(seat_ath, i32),
        seat_weight=jnp.asarray(np.full(seat_boat.size, 0.5), f64),
        year0=jnp.asarray(2018.0, f64),
        n_athletes=4,
        n_venues=2,
        n_classes=2,
        n_types=2,
        n_comps=4,
    )


def test_time_window_masks_cover_all_boats():
    years = np.array([2020, 2020, 2021, 2022, 2023, 2023], float)
    masks = block_ep.time_window_masks(years, width=2, step=1)
    assert len(masks) >= 2
    assert np.any(masks, axis=0).all()  # every boat in at least one window
    assert any(m.sum() < len(years) for m in masks)  # windows actually partition


def test_athlete_windows_finds_shared():
    mi = _synthetic_mi()
    masks = block_ep.time_window_masks(np.asarray(mi.year), width=1.0, step=1.0)  # one window per year
    aw, shared = block_ep.athlete_windows(mi, masks)
    # every athlete races in several years -> all are shared across windows
    assert shared == {0, 1, 2, 3}


def test_one_window_equals_exact_gp():
    mi = _synthetic_mi()
    t_ref = 2023.0

    full_mask = np.ones(mi.n_boats, bool)
    mean, var, shared = block_ep.block_athlete_scores(mi, [full_mask], t_ref)
    assert shared == set()  # single window -> no cross-window athletes

    gp = PerformanceGP.from_inputs(mi)
    y_ath, cov_ath = gp.predict_athletes_score(t_ref)

    assert np.allclose(mean, np.asarray(y_ath), atol=1e-9)
    assert np.allclose(var, np.asarray(cov_ath).diagonal(), atol=1e-9)


def test_multi_window_runs_and_is_finite():
    mi = _synthetic_mi()
    t_ref = 2023.0
    masks = block_ep.time_window_masks(np.asarray(mi.year), width=2.0, step=1.0)
    assert len(masks) >= 2

    mean, var, shared = block_ep.block_athlete_scores(mi, masks, t_ref)
    assert len(shared) > 0
    assert np.isfinite(mean).all() and (var > 0).all()


def test_run_ep_one_window_equals_exact():
    mi = _synthetic_mi()
    t_ref = 2023.0
    full_mask = np.ones(mi.n_boats, bool)
    res = block_ep.run_ep(mi, [full_mask], t_ref, n_iter=3, damping=1.0)

    gp = PerformanceGP.from_inputs(mi)
    y_ath, cov_ath = gp.predict_athletes_score(t_ref)
    assert np.allclose(res.athlete_mean, np.asarray(y_ath), atol=1e-7)
    assert np.allclose(res.athlete_var, np.asarray(cov_ath).diagonal(), atol=1e-7)

    # boat-class baseline: exact = boatclass_var * (W_bc^T @ system.a)
    system = gp.gp_system()
    boatclass_var = float(np.asarray(gp.boatclass_var.value).reshape(()))
    W_bc = np.asarray(mi.one_hot("class"))
    bc_exact = boatclass_var * (W_bc.T @ np.asarray(system.a))
    assert np.allclose(res.class_mean, bc_exact, atol=1e-7)


def test_run_ep_converges_and_beats_poe():
    mi = _synthetic_mi()
    t_ref = 2023.0
    masks = block_ep.time_window_masks(np.asarray(mi.year), width=2.0, step=1.0)

    gp = PerformanceGP.from_inputs(mi)
    exact = np.asarray(gp.predict_athletes_score(t_ref)[0])

    poe_mean, _, shared = block_ep.block_athlete_scores(mi, masks, t_ref)
    res = block_ep.run_ep(mi, masks, t_ref, n_iter=25, damping=0.5)

    idx = np.array(sorted(shared))
    poe_err = np.abs(poe_mean[idx] - exact[idx]).max()
    ep_err = np.abs(res.athlete_mean[idx] - exact[idx]).max()
    print(f"\nPoE max err {poe_err:.4g} | EP max err {ep_err:.4g} | EP converged to {res.history[-1]:.2e}")

    assert np.isfinite(res.athlete_mean).all() and (res.athlete_var > 0).all()
    assert res.history[-1] < 1e-3  # converged
    assert ep_err <= poe_err + 1e-9  # EP at least as accurate as naive product-of-experts


def test_predict_boats_matches_simple_model():
    import pandas as pd

    from rowing.model.block_ep import EPResult

    mi = _synthetic_mi()
    t_ref = 2023.0
    masks = block_ep.time_window_masks(np.asarray(mi.year), width=2.0, step=1.0)

    # target competition: two boats drawn from the model's athletes, one per boat class
    comp_athletes = pd.DataFrame(
        {
            "boatId": ["A", "A", "B", "B"],
            "personId": [0, 1, 2, 3],
            "athletePosition": ["b", "s", "b", "s"],
        }
    )
    athlete_index = pd.Index([0, 1, 2, 3])
    boat_class = pd.Series({"A": 0, "B": 1})

    # exact simple-model posteriors -> EPResult-shaped, for the same boat-scoring path
    gp = PerformanceGP.from_inputs(mi)
    y_ath, cov_ath = gp.predict_athletes_score(t_ref)
    system = gp.gp_system()
    boatclass_var = float(np.asarray(gp.boatclass_var.value).reshape(()))
    bc_exact = boatclass_var * (np.asarray(mi.one_hot("class")).T @ np.asarray(system.a))
    exact = EPResult(np.asarray(y_ath), np.asarray(cov_ath).diagonal(), bc_exact, None, set(), [])

    ep = block_ep.run_ep(mi, masks, t_ref)

    yb_exact, _ = block_ep.predict_boats(exact, comp_athletes, athlete_index, boat_class)
    yb_ep, _ = block_ep.predict_boats(ep, comp_athletes, athlete_index, boat_class)

    assert list(yb_ep.index) == ["A", "B"]
    assert np.allclose(yb_ep.values, yb_exact.values, atol=0.15)  # within the multi-window EP approximation
    # same predicted faster boat
    assert np.sign(yb_ep["A"] - yb_ep["B"]) == np.sign(yb_exact["A"] - yb_exact["B"])
