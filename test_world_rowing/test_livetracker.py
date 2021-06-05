

import pytest

from world_rowing import api, livetracker


def test_livetracker():
    race = api.get_last_race_started()
    tracker = livetracker.RaceTracker(race.name)
    live_data = tracker.update_livedata()
    assert (live_data.distanceTravelled.iloc[-1] == 2000).all()
