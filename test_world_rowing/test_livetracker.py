

import pytest

from rowing.world_rowing import api, livetracker

def get_2021_olympics():
    return api.get_competitions(2021).loc["e807bba5-6475-4f1a-9434-26704585bf19"]


def test_livetracker():
    competition = get_2021_olympics()
    race = api.get_last_race_started(competition=competition)
    tracker = livetracker.RaceTracker(race.name)
    live_data, intermediates = tracker.update_livedata()
    # assert (live_data.distanceTravelled.iloc[-1] == 2000).all()
