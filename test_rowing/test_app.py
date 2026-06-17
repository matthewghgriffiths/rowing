"""Smoke tests for the Streamlit apps using the streamlit.testing AppTest framework.

https://docs.streamlit.io/develop/api-reference/app-testing

Each app's ``main(params)`` entry point is exercised inside a simulated
Streamlit runtime via ``AppTest.from_function``. ``from_function`` runs the
*source* of the passed function as a standalone script, so we use a small
self-contained wrapper (``_run_app``) that imports the target app by name --
giving it back its module globals -- and forwards the preloaded data passed in
through ``kwargs``.

The World Rowing pages need the live API and are marked ``network`` (deselected
unless ``--run-network`` is passed); the gpx/telemetry apps run offline from the
sample-data fixtures.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from rowing.analysis import files, telemetry

dirpath = Path(__file__).resolve().parent

# gps.py / telemetry.py are the Streamlit entry points at the repository root;
# put it on sys.path so the wrapper below can import them by name.
sys.path.insert(0, str(dirpath.parent))

# The apps do real work (API calls, parsing, plotting), so allow generous time.
APP_TIMEOUT = 120


def _run_app(dotted, params):
    """Self-contained AppTest script: import ``module:function`` and call it.

    Runs as a standalone script under AppTest, so it must do its own imports.
    """
    import importlib

    module_name, func_name = dotted.split(":")
    module = importlib.import_module(module_name)
    getattr(module, func_name)(params)


def run_app(dotted, params, timeout=APP_TIMEOUT):
    at = AppTest.from_function(
        _run_app, kwargs={"dotted": dotted, "params": params}, default_timeout=timeout
    )
    at.run()
    return at


@pytest.mark.network
@pytest.mark.parametrize(
    "params",
    [
        None,
        {"current_competition": True, "pickCBT": True, "results.modal": True, "intermediate_results.modal": False},
        {"current_competition": False, "pickCBT": False, "results.modal": False, "intermediate_results.modal": False},
        {
            "current_competition": False,
            "pickCBT": False,
            "results.modal": False,
            "intermediate_results.modal": False,
            "competition.modal": True,
            "GMT.modal": False,
        },
    ],
)
def test_GMTs(params):
    at = run_app("rowing.world_rowing.pages.pgmts:main", params)
    assert not at.exception


@pytest.mark.network
@pytest.mark.parametrize(
    "params",
    [
        None,
        {"current_competition": True, "filter_races.modal": True, "pickCBT": True, "live_data.modal": True},
        {"current_competition": False, "pickCBT": False, "live_data.modal": False},
        {
            "current_competition": False,
            "filter_races.modal": True,
            "pickCBT": False,
            "live_data.modal": False,
            "competition.modal": True,
            "GMT.modal": False,
        },
    ],
)
def test_livetracker(params):
    at = run_app("rowing.world_rowing.pages.livetracker:main", params)
    assert not at.exception


@pytest.mark.network
@pytest.mark.parametrize(
    "params",
    [
        {"replay": 50, "replay_step": 50, "replay_race": False},
    ],
)
def test_realtime(params):
    at = run_app("rowing.world_rowing.pages.realtime:main", params)
    assert not at.exception


def test_gpx(cam_gpx):
    gpx_data = {"cam": files.read_gpx(cam_gpx)}
    at = run_app("gps:main", {"gpx_data": gpx_data})
    assert not at.exception
    # The app rendered its page (title is set before any data processing).
    assert any("Rowing GPS Analysis" in t.value for t in at.title)


def test_telemetry(powerline_txt):
    telemetry_data = {"powerline": telemetry.parse_powerline_text_data(powerline_txt.read_text())}
    params = {
        "telemetry_data": telemetry_data,
        "Make profile plots": True,
        "Make all plots": True,
    }
    at = run_app("telemetry:main", params)
    assert not at.exception


def _run_world_rowing_home():
    import world_rowing

    world_rowing.main()


def test_world_rowing_home():
    # The landing page is pure markdown/image, so it runs offline.
    at = AppTest.from_function(_run_world_rowing_home, default_timeout=APP_TIMEOUT)
    at.run()
    assert not at.exception
    assert any("World Rowing" in t.value for t in at.title)


@pytest.mark.network
def test_results():
    at = run_app("rowing.world_rowing.pages.results:main", None)
    assert not at.exception


@pytest.mark.network
def test_entries():
    at = run_app("rowing.world_rowing.pages.entries:main", None)
    assert not at.exception
