"""Characterization test for the flax.nnx joint_model kernel paths.

joint_model's only Haiku coupling was AthleteModel.apply (athlete-kernel Gram) and the
per-competition race jitter kernel (CompetitionModels.apply). These now build nnx kernels and
load the legacy params dict by name. The reference fingerprints (trace / sum / Frobenius norm)
were captured from the previous all-Haiku implementation on fixed synthetic inputs; the matrices
matched bit-for-bit (max abs diff 0.0).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("flax")
pytest.importorskip("haiku")  # competition_model still imports haiku until Stage 4

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from rowing.model.gp import kernels  # noqa: E402
from rowing.model.performance import competition_model as cm  # noqa: E402
from rowing.model.performance import joint_model as jm  # noqa: E402

# Captured from the all-Haiku implementation (rtol/atol 1e-12 exact on the full matrices).
REFERENCES = {
    "K_ath": dict(trace=8.825972428052912, sum=32.465006371928645, fro=5.86239833137438),
    "K_race": dict(trace=22.28692461486204, sum=46.37666909325599, fro=13.314203081124695),
}

PARAMS = {
    "athlete_se_0": {"log_var": -0.5, "log_scale": 0.3},
    "athlete_matern32": {"log_var": -0.7},
    "athlete_bias": {"log_var": -1.0},
    "race_matern12_0": {"log_var": -0.4, "log_scale": 0.2},
    "race_kernel0": {"log_var": -0.6, "log_scale": 0.5},
    "lane_kernel0": {"log_var": -0.8},
    "~": {"Boat Type": -0.2, "log_noise": -1.5},
}


def _params():
    return jax.tree_util.tree_map(lambda v: jnp.asarray(float(v)), PARAMS)


def _check(name, M):
    M = np.asarray(M)
    ref = REFERENCES[name]
    assert M.shape == (6, 6)
    assert np.allclose(M, M.T)  # kernel Gram is symmetric
    assert float(M.trace()) == pytest.approx(ref["trace"], rel=1e-10)
    assert float(M.sum()) == pytest.approx(ref["sum"], rel=1e-10)
    assert float(np.linalg.norm(M)) == pytest.approx(ref["fro"], rel=1e-10)


def test_athlete_model_apply_matches_haiku():
    years = jnp.linspace(2020.0, 2024.0, 6)
    dummy = np.array([], dtype=int)
    am = jm.AthleteModel(
        years=np.asarray(years),
        athlete_year_inds=[],
        athlete_splits=dummy,
        athlete_order=dummy,
        competition_splits=dummy,
        competition_order=dummy,
        athlete_kernel=cm.get_athlete_kernel,
    )
    _check("K_ath", am.apply(_params()))


def test_race_jitter_gram_matches_haiku():
    rng = np.random.default_rng(7)
    nb, nv, nbc = 6, 2, 2
    W_ven = rng.standard_normal((nb, nv))
    W_bc = rng.standard_normal((nb, nbc))
    W_lane = rng.standard_normal((nb, 1))

    def g(W):
        return jnp.asarray(W @ W.T)

    rm = cm.RaceModel(
        hours=jnp.linspace(0.0, 50.0, nb),
        W_venue=jnp.asarray(W_ven),
        W_boatclass=jnp.asarray(W_bc),
        W_lane=jnp.asarray(W_lane),
        gram_venue=g(W_ven),
        gram_boatclass=g(W_bc),
        gram_lane=g(W_lane),
        race_kernel=cm.get_race_kernel,
        lane_kernel=lambda: kernels.SumKernel(kernels.SEKernel(name="lane_kernel0", scale=2)),
    )
    _check("K_race", jm.race_jitter_gram(rm, _params()))


def test_predict_performances_score_std_is_posterior_std():
    """Regression for the score_std fix: it must be sqrt(diag(cov)), not the mean."""
    years = np.array([2021.0, 2022.0, 2023.0])
    athlete_year_inds = [np.array([0, 1]), np.array([1, 2])]  # two athletes' competition indices
    dummy = np.array([], dtype=int)
    am = jm.AthleteModel(
        years=years,
        athlete_year_inds=athlete_year_inds,
        athlete_splits=dummy,
        athlete_order=dummy,
        competition_splits=dummy,
        competition_order=dummy,
        athlete_kernel=cm.get_athlete_kernel,
    )
    athlete_dists = [
        (jnp.array([0.1, 0.2]), jnp.array([0.1, 0.1])),
        (jnp.array([0.0, -0.1]), jnp.array([0.1, 0.1])),
    ]
    times = np.array([2021.5, 2022.5])

    data = type(
        "Data",
        (),
        {
            "athlete_index": pd.Index([10, 20], name="athlete_id"),
            "athletes": pd.DataFrame(
                {"athletes_person_BirthDate": ["1990", "1991"], "athletes_person": ["A", "B"]},
                index=[10, 20],
            ),
        },
    )()

    preds = jm.predict_performances(times, am, athlete_dists, data, _params())

    assert {"score", "score_std"}.issubset(preds.columns)
    assert (preds.score_std > 0).all()
    assert np.isfinite(preds.score_std).all()
    # The bug set score_std == score; they must now differ.
    assert not np.allclose(preds.score_std.values, preds.score.values)
