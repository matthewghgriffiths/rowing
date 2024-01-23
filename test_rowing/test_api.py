

import pytest

from rowing.world_rowing import api, fields

from rowing.utils import timeout 

TIMEOUT = 30

@timeout(TIMEOUT)
def test_data_retrieval():
    competitions = api.get_competitions()
    competition = api.get_most_recent_competition()

    # Load 2021 olympics for consistency
    competitions = api.get_competitions(2021)
    competition = competitions[
        competitions.competition_id == "e807bba5-6475-4f1a-9434-26704585bf19"
    ].iloc[0]

    events = api.get_events(competition.competition_id)
    races = api.get_races(competition.competition_id)

    race = races.iloc[1]
    event = events.iloc[0]

    race_result = api.get_race_results(
        race_id=race.race_id
    )
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
    competition = competitions[
        competitions.competition_id == "e807bba5-6475-4f1a-9434-26704585bf19"
    ].iloc[0]

    api.get_last_races(competition=competition)
    api.get_next_races(competition=competition)
    api.get_boat_classes()
    api.get_competition_types()
    api.get_statistics()
    api.get_venues()
    api.get_competition_best_times()
    api.get_world_best_times()
