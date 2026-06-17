"""Characterization test for the flax.nnx PerformanceGP.

References captured from the previous Haiku competition_model on a fixed synthetic
PerformanceModel (built directly from arrays, no data pipeline). Pins loss, the
full-kernel trace and the GP-system ``a`` vector at zero-init and all-params=0.5.
"""
import numpy as np
import pytest

pytest.importorskip("flax")
pytest.importorskip("haiku")  # competition_model still imports haiku until Stage 4

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import flax.nnx as nnx  # noqa: E402

from rowing.model.performance import competition_model as cm  # noqa: E402


REFERENCES = {
    "zero": dict(
        loss=7.598540277779008,
        full_kernel_trace=76.60475594936162,
        a=[-0.35947665211498575, 0.11139489815533392, 0.038243598924266806, 0.008721383681979868, -0.20836035114323687, -0.13447988311146755],
    ),
    "set": dict(
        loss=8.900675223989543,
        full_kernel_trace=126.29988654178416,
        a=[-0.22227928081186155, 0.07601944884182073, 0.022643684857092602, 0.004272518698356887, -0.13383162069768073, -0.08499523574661559],
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
