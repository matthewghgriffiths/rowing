"""Offline regression tests for rowing.app.inputs.filter_dataframe.

These stub out streamlit entirely -- they assert the shape of the st.data_editor call and the
frame that comes back, not any rendering.
"""

import contextlib

import pandas as pd
import pytest

from rowing.app import inputs


class FakeEditor:
    """Records the st.data_editor call and echoes the frame back, as streamlit does."""

    def __init__(self):
        self.calls = []

    def __call__(self, data, **kwargs):
        self.calls.append((data, kwargs))
        return data.copy()


class FakeStreamlit:
    def __init__(self, editor, checkbox=True):
        self.data_editor = editor
        self._checkbox = checkbox
        self.session_state = {}

    def container(self):
        return contextlib.nullcontext()

    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        return [self for _ in range(n)]

    def multiselect(self, *args, **kwargs):
        return []

    def checkbox(self, *args, **kwargs):
        return self._checkbox

    def dataframe(self, *args, **kwargs):
        return None


@pytest.fixture
def activities():
    return pd.DataFrame(
        {
            "activity": ["a", "b", "c"],
            "distance": [1000, 2000, 3000],
            "hidden": ["x", "y", "z"],
        }
    )


def _patch(monkeypatch, editor, checkbox=True):
    monkeypatch.setattr(inputs, "st", FakeStreamlit(editor, checkbox=checkbox))
    monkeypatch.setattr(inputs, "modal_button", lambda *a, **k: True)


def test_column_order_keeps_the_select_column_visible(monkeypatch, activities):
    # The selection lives in an ordinary "select" column, so it has to be named in the
    # column_order handed to data_editor -- otherwise the checkbox column is hidden and the
    # user cannot (de)select any row. Callers (garmin/strava) pass a column_order that lists
    # data columns only.
    editor = FakeEditor()
    _patch(monkeypatch, editor)

    inputs.filter_dataframe(
        activities,
        key="test",
        column_order=["activity", "distance"],
    )

    data, kwargs = editor.calls[0]
    assert kwargs["column_order"][0] == "select"
    assert kwargs["column_order"] == ["select", "activity", "distance"]
    # the select column must also be present in the frame passed to the editor, exactly once
    assert list(data.columns).count("select") == 1


def test_select_column_not_duplicated_when_already_in_column_order(monkeypatch, activities):
    editor = FakeEditor()
    _patch(monkeypatch, editor)

    inputs.filter_dataframe(
        activities,
        key="test",
        column_order=["select", "activity"],
    )

    data, kwargs = editor.calls[0]
    assert kwargs["column_order"] == ["select", "activity"]
    assert list(data.columns).count("select") == 1


def test_selection_is_read_back_by_name(monkeypatch, activities):
    # Row selection is boolean masking on the returned "select" column, not positional indexing.
    class PickMiddle(FakeEditor):
        def __call__(self, data, **kwargs):
            out = super().__call__(data, **kwargs)
            out["select"] = [False, True, False]
            return out

    editor = PickMiddle()
    _patch(monkeypatch, editor)

    out = inputs.filter_dataframe(activities, key="test")

    assert list(out["activity"]) == ["b"]
    assert list(out.index) == [1]


def test_no_column_order_leaves_it_unset(monkeypatch, activities):
    editor = FakeEditor()
    _patch(monkeypatch, editor)

    inputs.filter_dataframe(activities, key="test")

    _, kwargs = editor.calls[0]
    assert kwargs["column_order"] is None
