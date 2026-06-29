import pytest

from rowing.utils import timeout
from rowing.world_rowing import api, live

TIMEOUT = 60

# Every test in this module hits the live World Rowing API.
pytestmark = pytest.mark.network


@timeout(TIMEOUT)
def test_data_retrieval():
    competitions = api.get_competitions()
    competition = api.get_most_recent_competition()

    # Load 2021 olympics for consistency
    competitions = api.get_competitions(2021)
    competition = competitions[competitions.competition_id == "e807bba5-6475-4f1a-9434-26704585bf19"].iloc[0]

    events = api.get_events(competition.competition_id)
    races = api.get_races(competition.competition_id)

    race = races.iloc[1]
    event = events.iloc[0]

    race_result = api.get_race_results(race_id=race.race_id)
    api.get_race_results(event_id=event.event_id)
    api.get_race_results(competition_id=competition.competition_id)
    api.get_intermediate_results(race_id=race.race_id)
    api.get_intermediate_results(event_id=event.event_id)
    api.get_intermediate_results(competition_id=competition.competition_id)
    api.find_world_best_time(race_id=race.race_id)


@timeout(TIMEOUT)
def test_get_stats():
    # Load 2021 olympics for consistency
    competitions = api.get_competitions(2021)
    competition = competitions[competitions.competition_id == "e807bba5-6475-4f1a-9434-26704585bf19"].iloc[0]

    api.get_last_races(competition=competition)
    api.get_next_races(competition=competition)
    api.get_boat_classes()
    api.get_competition_types()
    api.get_statistics()
    api.get_venues()
    api.get_competition_best_times()
    api.get_world_best_times()


@timeout(TIMEOUT)
def test_livetracker_pipeline():
    # End-to-end livetracker path used by the livetracker page: load_livetracker (request plumbing)
    # + estimate_livetracker_times (numeric coercion of the "d500m" distance labels). Catches both
    # the request_worldrowing params regression and the numeric-vs-str comparison regression.
    competitions = api.get_competitions(2021)
    competition = competitions[competitions.competition_id == "e807bba5-6475-4f1a-9434-26704585bf19"].iloc[0]
    races = api.get_races(competition.competition_id)
    race_ids = races.race_id.iloc[:8]

    # Direct estimate on a real race -> raises if estimate_livetracker_times regresses (the batch
    # path below swallows per-race errors, so test the function directly for a strong guard).
    for race_id in race_ids:
        live_boat_data, intermediates, lane_info, race_distance = live.load_livetracker(race_id, cached=False)
        if not live_boat_data.empty:
            out = live.estimate_livetracker_times(live_boat_data, intermediates, lane_info, race_distance)
            live_data = out[0] if isinstance(out, tuple) else out
            assert len(live_data) > 0
            break
    else:
        pytest.skip("no livetracker data for the sampled races")

    # batch path used by the app
    live_data, intermediates, lane_info = live.get_races_livetracks(race_ids)
    assert not live_data.empty
