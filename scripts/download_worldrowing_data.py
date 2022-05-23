
from collections import Counter

import pandas as pd
import numpy as np

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from rowing.world_rowing import api, livetracker, utils

data_store = 'live_tracker.h5'


def main():
    races, events, competitions = load_races(
            year_start=2008, year_end=2022, 
    )
    races = merge_race_event_competitions(races, events, competitions)
    race_results = load_competition_results(competitions.index)
    race_live_data = load_livedata(races.index)

    with pd.HDFStore(data_store) as store:
        store['races'] = races 
        store['race_results'] = race_results
        for race_id, livetrack in race_live_data.items():
            live_data, results, intermediates = livetrack
            if len(live_data):
                store['livetracker/live_data/' + race_id] = live_data
                # store['livetracker/results/' + race_id] = results
                # store['livetracker/intermediates/' + race_id] = intermediates



def load_races(
        year_start=2008, year_end=2022, 
):

    competitions = api.get_worldrowing_records(
        'competition', 
        filter=(
            ('Year', tuple(range(year_start, year_end))),
            ('IsFisa', 1)
        )
    )
    competition_races, errors = utils.map_concurrent(
        api.get_competition_races,
        dict(
            zip(
                competitions.index, 
                zip(competitions.index)
            )
        )
    )
    races = pd.concat(
        competition_races, axis=0,
        names=['competitionId', 'id']
    ).reset_index(0)

    competition_events, errors = utils.map_concurrent(
        api.get_competition_events,
        dict(
            zip(
                competitions.index, 
                zip(competitions.index)
            )
        )
    )
    events = pd.concat(
        competition_events, axis=0,
        names=['competitionId', 'id']
    ).reset_index(0, drop=True)

    # Set string column types
    for df in [races, events, competitions]:
        for col, dtype in df.dtypes.items():
            df[col] = df[col].astype(
                str if dtype == np.dtype('O') else dtype
            )


    return races, events, competitions
    
def merge_race_event_competitions(races, events, competitions):
    return utils.merge(
        (
            races.reset_index(), events, 
            competitions, api.get_boat_types()
        ),
        how='left',
        left_on=('eventId', 'competitionId', 'boatClassId'),
        right_on='id',
        suffixes=(
            (None, '_event'),
            (None, '_competition'),
            (None, '_boat_class')
        )
    ).set_index('id')

def get_competition_results(comp_id):
    return api.get_race_results(competition_id=comp_id)

def load_competition_results(competition_ids):
    comp_results, errors = utils.map_concurrent(
        get_competition_results, 
        {c: (c,) for c in competition_ids}
    )
    race_results = pd.concat({
        k: df for k, df in comp_results.items() if len(df)
    }, 
        names = ['competition_id', 'race_id', 'id'],
        axis=0).reset_index(0)

    return race_results

def load_livedata(race_ids):
    race_live_data, errors = utils.map_concurrent(
        livetracker.get_race_livetracker,
        {rid: (rid,) for rid in race_ids},
        max_workers=30,
    )
    return race_live_data

def save_livedata(race_live_data, data_store):
    with pd.HDFStore(data_store) as store:
        for race_id, livetrack in race_live_data.items():
            live_data, results, intermediates = livetrack
            if len(live_data):
                store['livetracker/live_data/' + race_id] = live_data
                store['livetracker/results/' + race_id] = results
                store['livetracker/intermediates/' + race_id] = intermediates


if __name__ == "__main__":
    main()