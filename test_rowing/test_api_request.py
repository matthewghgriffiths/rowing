"""Offline regression tests for the request plumbing in rowing.world_rowing.api.

The live-API tests in test_api.py are network-gated, so a bug in how the shared session is
*called* (e.g. passing params positionally) would never be caught offline. These tests stub the
session and assert the call shape only -- no network.
"""

from rowing.world_rowing import api


def test_request_worldrowing_passes_params_as_keyword(monkeypatch):
    # FakeSession.get mirrors requests.Session.get: only `url` is positional. If request_worldrowing
    # regresses to _session.get(url, params) the positional params raises TypeError here.
    calls = {}

    class FakeSession:
        def get(self, url, **kwargs):
            calls["url"] = url
            calls["kwargs"] = kwargs
            return "RESPONSE"

    monkeypatch.setattr(api, "_session", FakeSession())

    out = api.request_worldrowing("livetracker", "race-123", competition="abc")

    assert out == "RESPONSE"
    assert calls["url"].endswith("/livetracker/race-123")
    assert "params" in calls["kwargs"]  # params must be a keyword, not the 2nd positional arg


def test_prepare_request_builds_url_and_params():
    url, params = api.prepare_request("livetracker", "race-123")
    assert url.endswith("/stats/api/livetracker/race-123")
    assert params is None  # no query kwargs -> None (so Session.get omits params)
