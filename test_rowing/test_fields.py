"""Unit tests for rowing.world_rowing.fields (column naming and dtype mapping)."""
import numpy as np
import pandas as pd

from rowing.world_rowing import fields


def test_field_names_casefolded_index_is_consistent():
    # FIELD_NAMES is the casefolded lookup built from field_names.
    assert all(key == key.casefold() for key in fields.FIELD_NAMES)
    assert fields.FIELD_NAMES["pgmt"] == "PGMT"


def test_rename_column_known_prefixed():
    assert fields.rename_column("Date", prefix="race") == "Race Start"
    assert fields.rename_column("id", prefix="race") == "race_id"


def test_rename_column_unknown_falls_back_to_joined_name():
    assert fields.rename_column("zzz", prefix="foo") == "foo_zzz"


def test_renamer_is_partial_of_rename_column():
    rename_race = fields.renamer("race")
    assert rename_race("Date") == "Race Start"


def test_which_dtype_classifies_common_dtypes():
    assert fields.which_dtype(np.dtype("int64")) == "numeric"
    assert fields.which_dtype(np.dtype("float64")) == "numeric"
    assert fields.which_dtype(np.dtype("O")) == "categorical"
    assert fields.which_dtype(pd.Series(pd.to_timedelta([1, 2], unit="s")).dtype) == "timedelta"
    assert fields.which_dtype(pd.Series(pd.to_datetime(["2021-01-01"])).dtype) == "datetime"


def test_filter_numerical_and_categorical_columns():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "a"],
            "when": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
        }
    )
    numerical = fields.filter_numerical_columns(df)
    categorical = fields.filter_categorical_columns(df)
    assert "num" in numerical
    assert "when" in numerical
    assert "cat" in categorical
