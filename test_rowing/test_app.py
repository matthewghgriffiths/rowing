import logging
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

import streamlit as st

from rowing.analysis import files, telemetry
from rowing.world_rowing import pages

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

dirpath = Path(__file__).resolve().parent

# gps.py / telemetry.py are the Streamlit entry points at the repository root,
# so make sure the repo root is importable regardless of the working directory.
sys.path.insert(0, str(dirpath.parent))
import gps as app_gpx
import telemetry as app_telemetry

TIMEOUT = 120


def run_streamlit(main, params):
    try:
        main(params)
    except (
        st.runtime.scriptrunner.StopException,
        st.runtime.scriptrunner.RerunException,
    ):
        pass


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
    run_streamlit(pages.pgmts.main, params)


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
    run_streamlit(pages.livetracker.main, params)


@pytest.mark.network
@pytest.mark.parametrize(
    "params",
    [
        {"replay": 50, "replay_step": 50, "replay_race": False},
    ],
)
def test_realtime(params):
    run_streamlit(pages.realtime.main, params)


def test_telemetry(powerline_txt):
    telemetry_data = {"powerline": telemetry.parse_powerline_text_data(powerline_txt.read_text())}
    params = {
        "telemetry_data": telemetry_data,
        "Make profile plots": True,
        "Make all plots": True,
    }
    run_streamlit(app_telemetry.main, params)


def test_gpx(cam_gpx):
    gpx_data = {
        "cam": files.read_gpx(cam_gpx),
    }
    params = {"gpx_data": gpx_data}
    run_streamlit(app_gpx.main, params)
