


import pytest

from rowing.world_rowing import api


def test_data_retrieval():
    competitions = api.get_competitions()
    competition = api.get_most_recent_competition()

    # Load 2021 olympics for consistency
    competition = api.get_competitions(2021).loc["e807bba5-6475-4f1a-9434-26704585bf19"]

    events = api.get_competition_events(competition.name) 
    races = api.get_competition_races(competition.name)

    race = races.iloc[1]

    race_result = api.get_race_results(
        race_id = race.name
    )
    api.get_race_results(
        event_id = events.index[0]
    )
    api.get_race_results(
        competition_id = competition.index[0]
    )
    api.get_intermediate_results(
        race_id = race.name
    )
    api.get_intermediate_results(
        event_id = events.index[0]
    )
    api.get_intermediate_results(
        competition_id = competition.index[0]
    )
    api.find_world_best_time(
        race_id = race.name
    )

def test_get_stats():
    # Load 2021 olympics for consistency
    competition = api.get_competitions(2021).loc["e807bba5-6475-4f1a-9434-26704585bf19"]

    api.get_last_races(competition=competition)
    api.get_next_races(competition=competition)
    api.get_boat_types()
    api.get_competition_types()
    api.get_statistics()
    api.get_venues()
    api.get_competition_best_times()
    api.get_world_best_times()



    