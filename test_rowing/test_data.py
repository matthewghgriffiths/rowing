"""Tests for the unified data layer (rowing.model.data).

Stage 1: ModelInputs + RowingData.to_inputs + PerformanceGP.from_inputs reproduce the pandas
PerformanceModel.from_data path exactly, on a small in-test synthetic dataset. Also exercises
ModelInputs.subset.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("flax")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from rowing.model.data import RowingData  # noqa: E402
from rowing.model.performance import competition_model as cm  # noqa: E402


def _synthetic_filtered():
    """A tiny filtered-results dict shaped like competition_model.filter_results output."""
    # 4 boats across 2 races (2 lanes each), crews of 2 athletes from {a, b, c}.
    boat_ids = [0, 1, 2, 3]
    results = pd.DataFrame(
        {
            "PGMT": [0.95, 0.93, 0.97, 0.91],
            "year": [2022.4, 2022.4, 2023.4, 2023.4],
            "Boat Type": ["M2-"] * 4,
            "Boat Class": ["M2-"] * 4,
            "race_event_competition_venueId": ["v1", "v1", "v2", "v2"],
            "race_event_competitionId": ["c1", "c1", "c2", "c2"],
            "Day": ["d1", "d1", "d2", "d2"],
            "Lane": [1, 2, 1, 2],
            "race_id": ["r1", "r1", "r2", "r2"],
            "Race Start": pd.to_datetime(["2023-06-01", "2023-06-01", "2023-06-02", "2023-06-02"]),
        },
        index=pd.Index(boat_ids, name="raceBoats_id"),
    )
    crews = {0: ["a", "b"], 1: ["a", "c"], 2: ["b", "c"], 3: ["a", "b"]}
    tuples = [(b, p) for b, ps in crews.items() for p in ps]
    seats = pd.DataFrame(
        {"athletes_boatPosition": ["b", "s"] * 4},
        index=pd.MultiIndex.from_tuples(tuples, names=["athletes_raceBoatId", "athletes_personId"]),
    )
    athletes = pd.DataFrame(index=pd.Index(["a", "b", "c"], name="athletes_personId"))
    competitions = pd.DataFrame({"competition_id": ["c1", "c2"]})
    return {"results": results, "seats": seats, "athletes": athletes, "competitions": competitions}


def test_from_inputs_reproduces_from_data():
    filtered = _synthetic_filtered()

    gp_data = cm.PerformanceGP(cm.PerformanceModel.from_data(**filtered))

    mi = RowingData(**filtered).to_inputs()
    gp_in = cm.PerformanceGP.from_inputs(mi)

    # Grams are the core of the model; they must match bit-for-bit.
    for attr in ("gram_athlete", "gram_venue", "gram_boatclass", "gram_lane"):
        assert np.allclose(np.asarray(getattr(gp_data, attr)), np.asarray(getattr(gp_in, attr)), atol=0, rtol=0), attr
    assert np.allclose(np.asarray(gp_data.years), np.asarray(gp_in.years))
    assert np.allclose(np.asarray(gp_data.hours), np.asarray(gp_in.hours))
    assert float(gp_data.loss()) == pytest.approx(float(gp_in.loss()), rel=1e-12)
    assert np.allclose(gp_data.gp_system().a, gp_in.gp_system().a, atol=1e-12)


def test_model_inputs_shapes_and_grams():
    mi = RowingData(**_synthetic_filtered()).to_inputs()
    assert mi.n_boats == 4
    assert mi.n_athletes == 3
    assert mi.n_venues == 2 and mi.n_classes == 1 and mi.n_comps == 2
    # categorical Gram == one-hot @ one-hot.T
    W = mi.one_hot("venue")
    assert np.allclose(np.asarray(mi.categorical_gram("venue")), np.asarray(W @ W.T))
    # athlete Gram from seat scatter is symmetric PSD-ish and matches W_athlete @ W_athlete.T
    Wa = mi.W_athlete()
    assert np.allclose(np.asarray(mi.gram_athlete()), np.asarray(Wa @ Wa.T))


def test_model_inputs_subset():
    mi = RowingData(**_synthetic_filtered()).to_inputs()
    sub = mi.subset(jnp.array([0, 2]))  # keep boats 0 and 2

    assert sub.n_boats == 2
    assert sub.n_athletes == mi.n_athletes  # code cardinalities preserved
    assert np.allclose(np.asarray(sub.y), np.asarray(mi.y)[[0, 2]])
    # seats of boats 0 ({a,b}) and 2 ({b,c}) survive, reindexed to new boat positions 0,1
    assert set(np.asarray(sub.seat_boat)) <= {0, 1}
    assert len(sub.seat_boat) == 4
    # subset Gram equals the full Gram restricted to the kept boats
    full_g = np.asarray(mi.gram_athlete())[np.ix_([0, 2], [0, 2])]
    assert np.allclose(np.asarray(sub.gram_athlete()), full_g)


def test_predict_competition_smoke():
    rd = RowingData(**_synthetic_filtered())
    gp = rd.performance_gp()  # zero-init params

    # target competition: one event of two boats drawn from the training athletes
    boats = pd.DataFrame({"id": [0, 1], "Event": ["E1", "E1"], "Boat Type": ["M2-", "M2-"], "DisplayName": ["USA", "ITA"]})
    comp_athletes = pd.DataFrame(
        {
            "personId": ["a", "b", "a", "c"],
            "boatId": [0, 0, 1, 1],
            "athletePosition": ["b", "s", "b", "s"],
        }
    )

    out = rd.predict_competition(gp, boats, comp_athletes, start=2023.5, n_samples=2000, seed=0)

    assert set(out) >= {"y_boat", "cov_boat", "boat_class", "athlete_scores", "exp_score", "event_ranks"}
    assert list(out["y_boat"].index) == [0, 1]
    assert np.isfinite(np.asarray(out["y_boat"])).all()
    # rank 1 scores 6 points .. rank 6 scores 1; a 2-boat event lands in [5, 6]
    assert out["exp_score"].between(5, 6).all()
    assert np.isfinite(np.asarray(out["athlete_scores"])).all()
