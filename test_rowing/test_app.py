
import importlib 
import logging

import pytest

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

app_GMTs = importlib.import_module("pages.1_GMTs", "..")
app_livetracker = importlib.import_module("pages.2_livetracker", "..")


def test_GMTs():
    app_GMTs.main()


def test_livetracker():
    app_livetracker.main()