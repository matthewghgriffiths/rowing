"""Characterization test for the flax.nnx PerformanceGP.

References captured from the previous Haiku competition_model on a fixed synthetic
PerformanceModel (built directly from arrays, no data pipeline). Pins loss, the
full-kernel trace and the GP-system ``a`` vector at zero-init and all-params=0.5.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flax")
pytest.importorskip("haiku")  # competition_model still imports haiku until Stage 4

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import flax.nnx as nnx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rowing.model.performance import competition_model as cm  # noqa: E402

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


REFERENCES = {
    "zero": dict(
        loss=7.598540277779008,
        full_kernel_trace=76.60475594936162,
        a=[
            -0.35947665211498575,
            0.11139489815533392,
            0.038243598924266806,
            0.008721383681979868,
            -0.20836035114323687,
            -0.13447988311146755,
        ],
    ),
    "set": dict(
        loss=8.900675223989543,
        full_kernel_trace=126.29988654178416,
        a=[
            -0.22227928081186155,
            0.07601944884182073,
            0.022643684857092602,
            0.004272518698356887,
            -0.13383162069768073,
            -0.08499523574661559,
        ],
    ),
}


def _synthetic_model():
    rng = np.random.default_rng(0)
    nb, na, nv, nbc = 6, 4, 2, 2

    def g(W):
        return jnp.asarray(W @ W.T)

    W_ath = rng.standard_normal((nb, na))
    W_ven = rng.standard_normal((nb, nv))
    W_bc = rng.standard_normal((nb, nbc))
    W_lane = rng.standard_normal((nb, 1))

    athlete_model = cm.AthleteModel(
        years=jnp.linspace(2020.0, 2024.0, nb),
        year0=2018.0,
        W_athlete=jnp.asarray(W_ath),
        gram_athlete=g(W_ath),
    )
    race_model = cm.RaceModel(
        hours=jnp.linspace(0.0, 100.0, nb),
        W_venue=jnp.asarray(W_ven),
        W_boatclass=jnp.asarray(W_bc),
        W_lane=jnp.asarray(W_lane),
        gram_venue=g(W_ven),
        gram_boatclass=g(W_bc),
        gram_lane=g(W_lane),
    )
    return cm.PerformanceModel(athlete_model=athlete_model, race_model=race_model, y=jnp.asarray(rng.standard_normal(nb)))


def _set_all_params(module, value):
    state = nnx.state(module, nnx.Param)
    state = jax.tree_util.tree_map(lambda x: jnp.full_like(x, value), state)
    nnx.update(module, state)


@pytest.mark.parametrize("setting", ["zero", "set"])
def test_performance_gp_matches_haiku(setting):
    ref = REFERENCES[setting]
    gp = cm.PerformanceGP(_synthetic_model())
    if setting == "set":
        _set_all_params(gp, 0.5)

    rtol = 1e-9 if setting == "zero" else 1e-4
    assert float(gp.loss()) == pytest.approx(ref["loss"], rel=rtol, abs=1e-7)
    assert float(jnp.trace(gp.get_full_kernel())) == pytest.approx(ref["full_kernel_trace"], rel=rtol, abs=1e-7)
    assert np.allclose(gp.gp_system().a, ref["a"], rtol=rtol, atol=1e-8)


def test_predict_athletes_scores_smoke():
    """Predict reuses the verified kernel + GP-system `a`; check shapes and consistency."""
    gp = cm.PerformanceGP(_synthetic_model())
    na = gp.W_athlete.shape[1]
    times = jnp.array([2021.0, 2023.0, 2025.0])

    scores = gp.predict_athletes_scores(times)
    assert scores.shape == (na, len(times))
    assert np.isfinite(np.asarray(scores)).all()

    # The single-time mean must equal the matching column of the multi-time predict.
    system = gp.gp_system()
    y_ath, cov_ath = gp.predict_athletes_score(2023.0, system=system)
    assert y_ath.shape == (na,)
    assert cov_ath.shape == (na, na)
    assert np.allclose(np.asarray(y_ath), np.asarray(scores[2023.0]))


def _canonical_model():
    """Synthetic PerformanceModel with the example pipeline's named kernels."""
    from rowing.model.gp import kernels as K

    rng = np.random.default_rng(1)
    nb, na, nv, nbc = 8, 5, 3, 2

    def g(W):
        return jnp.asarray(W @ W.T)

    W_ath = rng.standard_normal((nb, na))
    W_ven = rng.standard_normal((nb, nv))
    W_bc = rng.standard_normal((nb, nbc))
    W_lane = rng.standard_normal((nb, 1))

    athlete_model = cm.AthleteModel(
        years=jnp.linspace(2020.0, 2024.0, nb),
        year0=2018.0,
        W_athlete=jnp.asarray(W_ath),
        gram_athlete=g(W_ath),
        athlete_kernel=cm.get_athlete_kernel,  # SE[athlete_se_0]+Matern32[athlete_matern32]+Bias[athlete_bias]
    )
    race_model = cm.RaceModel(
        hours=jnp.linspace(0.0, 100.0, nb),
        W_venue=jnp.asarray(W_ven),
        W_boatclass=jnp.asarray(W_bc),
        W_lane=jnp.asarray(W_lane),
        gram_venue=g(W_ven),
        gram_boatclass=g(W_bc),
        gram_lane=g(W_lane),
        race_kernel=cm.get_race_kernel,  # Matern12[race_matern12_0]+SE[race_kernel0]
        lane_kernel=lambda: K.SumKernel(K.SEKernel(name="lane_kernel0", scale=2)),
    )
    return cm.PerformanceModel(athlete_model=athlete_model, race_model=race_model, y=jnp.asarray(rng.standard_normal(nb)))


def test_load_haiku_params_maps_names_onto_hypers():
    yaml = pytest.importorskip("yaml")
    params_path = EXAMPLES_DIR / "params.yaml"
    if not params_path.exists():
        pytest.skip("examples/params.yaml not present")
    params = yaml.safe_load(params_path.read_text())

    gp = cm.PerformanceGP(_canonical_model())
    cm.load_haiku_params(gp, params)

    se, matern32, bias = gp.athlete_kernel.kernels
    assert float(se.variance.value) == pytest.approx(np.exp(params["athlete_se_0"]["log_var"]))
    assert float(se.scale.value) == pytest.approx(np.exp(params["athlete_se_0"]["log_scale"]))
    assert float(matern32.variance.value) == pytest.approx(np.exp(params["athlete_matern32"]["log_var"]))
    assert float(matern32.scale.value) == pytest.approx(1.0)  # fixed at construction, yaml has no scale
    assert float(bias.variance.value) == pytest.approx(np.exp(params["athlete_bias"]["log_var"]))

    matern12, race_se = gp.race_kernel.kernels
    assert float(matern12.variance.value) == pytest.approx(np.exp(params["race_matern12_0"]["log_var"]))
    assert float(matern12.scale.value) == pytest.approx(np.exp(params["race_matern12_0"]["log_scale"]))
    assert float(race_se.variance.value) == pytest.approx(np.exp(params["race_kernel0"]["log_var"]))

    (lane_se,) = gp.lane_kernel.kernels
    assert float(lane_se.variance.value) == pytest.approx(np.exp(params["lane_kernel0"]["log_var"]))
    assert float(lane_se.scale.value) == pytest.approx(2.0)  # fixed; yaml's lane log_scale is ignored

    assert float(gp.boatclass_var.value) == pytest.approx(np.exp(params["~"]["Boat Type"]))
    assert float(gp.log_noise[...]) == pytest.approx(params["~"]["log_noise"])


def test_dump_load_round_trips():
    yaml = pytest.importorskip("yaml")
    params_path = EXAMPLES_DIR / "params.yaml"
    if not params_path.exists():
        pytest.skip("examples/params.yaml not present")
    params = yaml.safe_load(params_path.read_text())

    gp = cm.PerformanceGP(_canonical_model())
    cm.load_haiku_params(gp, params)
    dumped = cm.dump_haiku_params(gp)

    # reload the dumped params into a fresh model -> identical GP
    gp2 = cm.PerformanceGP(_canonical_model())
    cm.load_haiku_params(gp2, dumped)
    assert float(gp.loss()) == pytest.approx(float(gp2.loss()), rel=1e-12)
    assert np.allclose(gp.gp_system().a, gp2.gp_system().a, atol=1e-12)


def test_fit_robust_downweights_outlier():
    model = _synthetic_model()
    # inject a gross outlier into one boat's target
    y = np.asarray(model.y).copy()
    y[0] += 10.0
    model = model._replace(y=jnp.asarray(y))

    gp = cm.PerformanceGP(model)
    w = gp.fit_robust(nu=4.0, n_iter=10)

    assert w.shape == (len(y),)
    assert np.all(w > 0)
    assert w.argmin() == 0  # the injected outlier is the most down-weighted row
    assert w[0] < 0.5 * np.median(w)  # ... and strongly so
    # obs_weight feeds the jitter kernel -> larger effective noise on the outlier row
    assert float(gp.obs_weight[0]) == pytest.approx(w[0])


def test_fit_performance_gp_reduces_loss():
    """The nnx fit flow should reduce the PerformanceGP loss."""
    from rowing.model.gp.utils import fit_module

    gp = cm.PerformanceGP(_synthetic_model())
    before = float(gp.loss())
    res = fit_module(gp, options={"maxiter": 50})
    after = float(gp.loss())

    assert after <= before
    assert res["loss_history"][-1] == pytest.approx(after, abs=1e-6)
