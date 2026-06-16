"""Shared pytest configuration and fixtures for the rowing test suite.

Two things are set up here:

* a ``network`` marker that is skipped by default so the default test run is
  fast and offline.  Pass ``--run-network`` to exercise the live-API tests.
* fixtures that resolve the sample data files under ``<repo>/data`` and skip
  (rather than error) when a fixture file is absent.
"""

from pathlib import Path

import pytest

# ``test_rowing`` lives at the repository root, alongside ``data``.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.network (hit live external services)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="needs --run-network option to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.fixture(scope="session")
def data_dir():
    if not DATA_DIR.is_dir():
        pytest.skip(f"data directory not found at {DATA_DIR}")
    return DATA_DIR


@pytest.fixture(scope="session")
def cam_gpx(data_dir):
    path = data_dir / "cam.gpx"
    if not path.is_file():
        pytest.skip("cam.gpx fixture missing")
    return path


@pytest.fixture(scope="session")
def powerline_txt(data_dir):
    path = data_dir / "powerline.txt"
    if not path.is_file():
        pytest.skip("powerline.txt fixture missing")
    return path
