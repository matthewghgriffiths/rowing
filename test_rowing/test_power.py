"""Unit tests for rowing.analysis.power (max-mean power and piece detection)."""
import numpy as np
import pandas as pd

from rowing.analysis import power


def test_max_mean_power_constant_signal():
    t = np.arange(0, 120, 1.0)
    p = np.full_like(t, 250.0)
    intervals = np.array([0, 1, 2, 5, 10, 30, 60])

    best = power.max_mean_power(p, t, intervals)

    # Every windowed mean of a constant signal equals the constant.
    assert np.allclose(best.values, 250.0)


def test_max_mean_power_curve_is_non_increasing():
    # A decaying power trace: the best-effort curve must not increase with duration.
    rng = np.random.default_rng(0)
    t = np.arange(0, 300, 1.0)
    p = 400 - 0.5 * t + rng.normal(scale=5.0, size=t.size)
    intervals = np.array([0, 1, 5, 10, 30, 60, 120, 240])

    best = power.max_mean_power(p, t, intervals).sort_index()
    values = best.values.flatten()
    assert np.all(np.diff(values) <= 1e-9)


def _stroke_series(durations_s, power_w):
    """Build a stroke-power Series indexed by cumulative time in milliseconds."""
    times_ms = np.cumsum(np.r_[0.0, durations_s[:-1]]) * 1000.0
    return pd.Series(power_w, index=times_ms)


def test_identify_pieces_all_stopped_yields_no_piece():
    # 30s between strokes -> rating 2/min, below the default min_rating of 10.
    durations = np.full(20, 30.0)
    sp = _stroke_series(durations, np.full(20, 60.0))

    piece = power.identify_pieces(sp)
    assert (piece == 0).all()


def test_identify_pieces_detects_single_high_rate_block():
    # Warm-up (slow) -> a clear high-rate block -> cool-down (slow).
    durations = np.r_[np.full(8, 20.0), np.full(40, 3.0), np.full(8, 20.0)]
    sp = _stroke_series(durations, np.full(durations.size, 300.0))

    piece = power.identify_pieces(sp)

    # Exactly one contiguous piece is found (label > 0; stopped strokes are 0).
    positive = piece[piece > 0]
    assert positive.nunique() == 1
    # The smoothing window trims a few strokes off each end of the block.
    assert len(positive) > 10
