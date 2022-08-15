

import pytest

from rowing.world_rowing import dashboard, api

def get_2021_olympics():
    return api.get_competitions(2021).loc["e807bba5-6475-4f1a-9434-26704585bf19"]

def test_dashboard_main():
    competition = get_2021_olympics()
    dash = dashboard.Dashboard.load_last_race(competition=competition)
    dash.update()
    # dashboard.main(block=False)

def test_dashboard_predict():
    competition = get_2021_olympics()
    dash = dashboard.Dashboard.load_last_race(competition=competition)
    live_data, intermediates = dash.race_tracker.update_livedata()

    dash.update(
        live_data.loc[:300],
        intermediates.loc[[500, 1000]]
    )

    dash.update(
        live_data,
        intermediates
    )


