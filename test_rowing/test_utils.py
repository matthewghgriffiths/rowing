"""Unit tests for rowing.utils (interpolation, formatting, concurrency)."""

from datetime import timedelta

import pandas as pd

from rowing import utils


def test_interpolate_series_numeric():
    s = pd.Series([0.0, 10.0, 20.0], index=[0.0, 1.0, 2.0])
    out = utils.interpolate_series(s, [0.5, 1.5])
    assert out.tolist() == [5.0, 15.0]


def test_interpolate_timedelta():
    s = pd.Series(pd.to_timedelta([0, 10, 20], unit="s"), index=[0.0, 1.0, 2.0])
    out = utils.interpolate_timedelta(s, [0.5])
    assert out.dt.total_seconds().tolist() == [5.0]


def test_interpolate_datetime():
    base = pd.Timestamp("2021-01-01")
    s = pd.Series(
        [base, base + timedelta(seconds=10), base + timedelta(seconds=20)],
        index=[0.0, 1.0, 2.0],
    )
    out = utils.interpolate_datetime(s, [1.5])
    assert out.iloc[0] == base + timedelta(seconds=15)


def test_format_timedelta_and_totalseconds():
    assert utils.format_timedelta(timedelta(seconds=65)) == "1:05.00"
    assert utils.format_totalseconds(65, hundreths=False) == "1:05"
    assert utils.format_totalseconds(float("nan")) == ""


def test_timestamp_roundtrip():
    ts = pd.Timestamp("2021-06-01 12:00:00")
    out = utils.from_timestamp(utils.to_timestamp(ts))
    assert out == ts


def test_safe_name_and_initials():
    assert utils.safe_name("a:b:c") == "abc"
    assert utils.initials("Great Britain Eight") == "GBE"


def test_make_http_session_has_timeout_and_retries():
    from requests.adapters import HTTPAdapter

    session = utils.make_http_session(retries=5, backoff_factor=0.5, timeout=12)
    for scheme in ("https://example.com", "http://example.com"):
        adapter = session.get_adapter(scheme)
        assert isinstance(adapter, HTTPAdapter)
        assert adapter.max_retries.total == 5
        assert adapter.max_retries.backoff_factor == 0.5

    # The adapter injects the default timeout when a request omits it.
    adapter = session.get_adapter("https://example.com")
    captured = {}

    class _Resp:
        status_code = 200

    monkeypatched = HTTPAdapter.send
    HTTPAdapter.send = lambda self, request, **kw: captured.update(kw) or _Resp()
    try:
        adapter.send(request=object())
    finally:
        HTTPAdapter.send = monkeypatched
    assert captured["timeout"] == 12


def test_map_concurrent_returns_results():
    inputs = {i: (i,) for i in range(5)}
    output, errors = utils.map_concurrent(lambda x: x * 2, inputs, progress_bar=None)
    assert output == {i: i * 2 for i in range(5)}
    assert errors == {}


def test_map_concurrent_captures_errors():
    def boom(x):
        if x == 3:
            raise ValueError("nope")
        return x

    inputs = {i: (i,) for i in range(5)}
    output, errors = utils.map_concurrent(boom, inputs, progress_bar=None)
    assert set(output) == {0, 1, 2, 4}
    assert isinstance(errors[3], ValueError)
