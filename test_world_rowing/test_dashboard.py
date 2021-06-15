

import pytest

from world_rowing import dashboard 


def test_dashboard_main():
    dashboard.main(block=False)


def test_dashboard_predict():
    dash = dashboard.Dashboard.load_last_race()
    live_data, intermediates = dash.race_tracker.update_livedata()

    dash.update(
        live_data.loc[:300],
        intermediates[[500, 1000]]
    )

    dash.update(
        live_data,
        intermediates
    )


