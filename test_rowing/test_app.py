
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
        'current_competition': True, 'pickCBT': True, 'results.modal': True, 
        },
        {
        'current_competition': False, 'pickCBT': False, 'results.modal': False, 
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
            'None.modal': True,
            'pickCBT': True,
            'live_data.modal': True
        },
        {
            'current_competition': False, 
            'None.modal': False,
            'pickCBT': False,
            'live_data.modal': False
        },
    ]
)
def test_livetracker(params):
    app_livetracker.main(params)