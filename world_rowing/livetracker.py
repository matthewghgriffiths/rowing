
import numpy as np
import pandas as pd

from .api import get_worldrowing_data

def get_race_livetracker(race_id):
    data = get_worldrowing_data('livetracker', race_id)

    lane_boat = {
        lane['Lane']: lane for lane in data['config']['lanes']
    }
    rank_boat = {
        lane['Rank']: lane for lane in data['config']['lanes']
    }
    lane_cnt = {r: lane['DisplayName'] for r, lane in lane_boat.items()}
    rank_cnt = {r: lane['DisplayName'] for r, lane in rank_boat.items()}
    countries = [lane_cnt[i] for i in sorted(lane_cnt)]
    
    live_boat_data = {
        'currentPosition': {},
        'distanceTravelled': {},
        'distanceFromLeader': {},
        'strokeRate': {},
        'metrePerSecond': {},
    }
    for cnt in countries:
        for live_data in live_boat_data.values():
            live_data[cnt] = []

    for live_data in data['live']:
        for tracker in live_data['raceBoatTrackers']:
            cnt = lane_cnt[tracker['startPosition']]
            for key, live_data in live_boat_data.items():
                live_data[cnt].append(tracker[key])



    maxlen = max(
        max(map(len, live_data.values()))
        for live_data in live_boat_data.values()
    )
    for key, live_data in live_boat_data.items():
        for cnt, cnt_data in list(live_data.items()):
            cnt_len = len(cnt_data)
            if cnt_len == 0:
                del live_data[cnt]
            elif cnt_len < maxlen:
                cnt_data.extend(cnt_data[-1:] * (maxlen - cnt_len))

    live_boat_data = pd.concat(
        {
            key: pd.DataFrame.from_dict(live_data) 
            for key, live_data in live_boat_data.items()
        },
        axis=1
    )

    n_countries = len(live_boat_data.distanceTravelled.columns)
    # Estimate times for each distance
    boat_times = np.diff(
        np.c_[
            np.zeros(n_countries), 
            live_boat_data.distanceTravelled.values.T
        ], 
        axis=1
    ).T / live_boat_data.metrePerSecond
    for col in boat_times:
        live_boat_data['time', col] = boat_times[col].cumsum()

    return live_boat_data
    
    
    