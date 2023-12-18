
import importlib
import logging

import pytest

import streamlit as st
from rowing.app import select, inputs, state, plots

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


app_GMTs = importlib.import_module("app.world_rowing.pages.1_GMTs", "..")
app_livetracker = importlib.import_module("app.world_rowing.pages.2_livetracker", "..")
app_realtime = importlib.import_module("app.world_rowing.pages.3_realtime", "..")


def run_streamlit(main, params):
    try:
        main(params)
    except (
        st.runtime.scriptrunner.StopException,
        st.runtime.scriptrunner.RerunException,
    ):
        pass

@pytest.mark.parametrize(
    "params", [
        None,
        {
            'current_competition': True,
            'pickCBT': True,
            'results.modal': True,
            'intermediate_results.modal': False
        },
        {
            'current_competition': False,
            'pickCBT': False,
            'results.modal': False,
            'intermediate_results.modal': False
        },
        {
            'current_competition': False,
            'pickCBT': False,
            'results.modal': False,
            'intermediate_results.modal': False,
            'competition.modal': True,
            'GMT.modal': False,
        },
    ]
)
def test_GMTs(params):
    run_streamlit(app_GMTs.main, params)


@pytest.mark.parametrize(
    "params", [
        None,
        {
            'current_competition': True,
            'filter_races.modal': True,
            'pickCBT': True,
            'live_data.modal': True
        },
        {
            'current_competition': False,
            'pickCBT': False,
            'live_data.modal': False
        },
        {
            'current_competition': False,
            'filter_races.modal': True,
            'pickCBT': False,
            'live_data.modal': False,
            'competition.modal': True,
            'GMT.modal': False
        },
    ]
)
def test_livetracker(params):
    run_streamlit(app_livetracker.main, params)


@pytest.mark.parametrize(
    "params", [{'replay': 50, 'replay_step': 50, }, ]
)
def test_realtime(params):
    run_streamlit(app_realtime.main, params)
