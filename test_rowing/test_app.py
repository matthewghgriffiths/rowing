
import importlib 
import logging

import pytest

import streamlit as st
from rowing.app import select, inputs, state, plots

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


app_GMTs = importlib.import_module("pages.1_GMTs", "..")
app_livetracker = importlib.import_module("pages.2_livetracker", "..")

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
    app_GMTs.main(params)

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
    app_livetracker.main(params)