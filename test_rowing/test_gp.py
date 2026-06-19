"""Characterization tests for the flax.nnx Gaussian-process core.

The reference numbers were captured from the previous Haiku implementation on fixed
synthetic data (deterministic: all params zero-init, plus an all-params=0.5 setting), so
this pins the nnx port to the same numerics. Run with float64 enabled.
"""

import numpy as np
import pytest

pytest.importorskip("flax")

import jax

jax.config.update("jax_enable_x64", True)

import flax.nnx as nnx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rowing.model.gp import kernels as K  # noqa: E402
from rowing.model.gp.gpr import GaussianProcessRegression  # noqa: E402
from rowing.model.gp.utils import fit_module  # noqa: E402

T = jnp.linspace(0.0, 1.0, 8)
Y = jnp.sin(5 * T)
T1 = jnp.array([0.15, 0.55, 0.95])


# Reference outputs from the Haiku implementation (predict / predict_var / log_likelihood).
REFERENCES = {
    "se_zero": dict(
        predict=[0.3103960437138919, 0.045399406180225955, -0.2534726965175639],
        var=[0.17155297686250082, 0.12625862984119518, 0.20174883240843078],
        loglik=-25.727720973343153,
    ),
    "se_set": dict(
        predict=[0.2477987905429192, 0.09220018843568845, -0.04453498000205158],
        var=[0.2362402046055294, 0.19149471762759984, 0.26540600196104625],
        loglik=-27.847203455309028,
    ),
    "intse_zero": dict(
        predict=[-0.030131791843509975, -0.14305644974293993, -0.29010423919204753],
        var=[0.007581733950298059, 0.0831011753342274, 0.23149870458436195],
        loglik=-25.26789605585426,
    ),
    "intse_set": dict(
        predict=[0.7513179643574436, 0.4626516204319252, 0.15738359323625267],
        var=[0.11046655522936379, 0.002243407251925121, 0.18318586516233623],
        loglik=-26.778719175062314,
    ),
    "matern32_zero": dict(
        predict=[0.37071078778327116, 0.08756921037143738, -0.3620660843080056],
        var=[0.19781735228468933, 0.15714904936956076, 0.23436149210088264],
        loglik=-25.67647794097931,
    ),
    "matern32_set": dict(
        predict=[0.32948894195785317, 0.10110501382350268, -0.14186040396445443],
        var=[0.2729383699601522, 0.21343455177821014, 0.3153005131415245],
        loglik=-27.886025771064382,
    ),
    "sum_se_bias_zero": dict(
        predict=[0.3070261442434762, 0.04324777777082271, -0.25762047811939726],
        var=[0.1799780924045853, 0.1296932326018101, 0.2145124486089358],
        loglik=-26.21125352853103,
    ),
    "sum_se_bias_set": dict(
        predict=[0.22019863005367213, 0.06901734528641501, -0.0750286558807911],
        var=[0.24812300461975312, 0.19987828708906674, 0.2799109151156163],
        loglik=-28.41236238287898,
    ),
    "prod_se_matern_zero": dict(
        predict=[0.4274323456039538, 0.10975721130872168, -0.4405415210689819],
        var=[0.21240760961667715, 0.17195463161970448, 0.25445932371697244],
        loglik=-25.642832050763598,
    ),
    "prod_se_matern_set": dict(
        predict=[0.4389331743176811, 0.0869647228293613, -0.32454399459540084],
        var=[0.33737696555885766, 0.24609844454628815, 0.41094887305964],
        loglik=-28.426604090286666,
    ),
}


def _kernel(name):
    if name == "se":
        return K.SEKernel()
    if name == "intse":
        return K.IntSEKernel()
    if name == "matern32":
        return K.Matern32()
    if name == "sum_se_bias":
        return K.SumKernel(K.SEKernel(), K.Bias())
    if name == "prod_se_matern":
        return K.ProductKernel(K.SEKernel(), K.Matern32())
    raise ValueError(name)


def _set_all_params(module, value):
    state = nnx.state(module, nnx.Param)
    state = jax.tree_util.tree_map(lambda x: jnp.full_like(x, value), state)
    nnx.update(module, state)


@pytest.mark.parametrize("name", ["se", "intse", "matern32", "sum_se_bias", "prod_se_matern"])
@pytest.mark.parametrize("setting", ["zero", "set"])
def test_gp_matches_haiku_reference(name, setting):
    ref = REFERENCES[f"{name}_{setting}"]
    gp = GaussianProcessRegression(T, Y, kernel=_kernel(name))
    if setting == "set":
        _set_all_params(gp, 0.5)

    # zero-init is exact; the 0.5 setting carries minor float32/float64 param-dtype noise.
    rtol = 1e-9 if setting == "zero" else 1e-4

    pred = gp.predict(T1)
    _, var = gp.predict_var(T1)
    ll = float(gp.log_likelihood())

    assert np.allclose(pred, ref["predict"], rtol=rtol, atol=1e-9)
    assert np.allclose(var, ref["var"], rtol=rtol, atol=1e-9)
    assert ll == pytest.approx(ref["loglik"], rel=rtol, abs=1e-7)


def test_fit_module_improves_likelihood():
    """fit_module (nnx.split + scipy L-BFGS-B) should maximise the marginal likelihood."""
    gp = GaussianProcessRegression(T, Y, kernel=K.SEKernel())
    before = float(gp.log_likelihood())
    res = fit_module(gp)
    after = float(gp.log_likelihood())

    assert res.success
    assert after > before
    # With the marginal likelihood maximised and low obs-noise, the GP interpolates the data.
    assert np.allclose(np.asarray(gp.predict(T)), np.asarray(Y), atol=1e-3)
