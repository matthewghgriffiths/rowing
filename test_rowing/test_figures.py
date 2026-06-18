"""Unit tests for rowing.analysis.figures (pure figure builders/helpers)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rowing.analysis import figures


def test_outlier_range_orders_low_high():
    s = pd.Series(np.arange(0.0, 101.0))
    lo, hi = figures.outlier_range(s)
    assert lo < hi
    assert lo <= s.median() <= hi


def test_scatter_returns_figure_with_trace():
    df = pd.DataFrame({"x": [0, 1, 2], "y": [1.0, 2.0, 3.0]})
    fig = figures.scatter(df, "x", "y")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_scatter_adds_to_existing_figure():
    df = pd.DataFrame({"x": [0, 1, 2], "y": [1.0, 2.0, 3.0]})
    fig = figures.scatter(df, "x", "y")
    fig = figures.scatter(df, "x", "y", fig=fig)
    assert len(fig.data) == 2
