
import numpy as np
import pandas as pd

from .api import get_worldrowing_data, find_world_best_time

def get_race_livetracker(race_id, gmt=None, cached=True, race_distance=2000):
    data = get_worldrowing_data('livetracker', race_id, cached=cached)
    has_livedata = data and data['live']
    if not has_livedata:
        return pd.DataFrame([])

    gmt = gmt or find_world_best_time(
        race_id=race_id
    ).ResultTime.total_seconds()

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

    
    gmt_speed = race_distance / gmt
    countries = live_boat_data.time.columns
    for cnt in countries:
        live_boat_data['GMT', cnt] = \
            live_boat_data.distanceTravelled[cnt]/gmt_speed
    for cnt in countries:
        live_boat_data['PGMT', cnt] = \
            live_boat_data.GMT[cnt] / live_boat_data.time[cnt]


    return live_boat_data
    
    
def plot_livedata(live_data):
    import matplotlib.pyplot as plt

    f, axes = plt.subplots(3, figsize=(10, 8), sharex=True)
    countries = live_data.time.columns
    lines = [[] for _ in range(3)]
    for c in countries:
        lines[0].extend(
            axes[0].plot(
            live_data.distanceTravelled[c], 
            live_data.PGMT[c], 
            label=c
            )
        )
        lines[1].extend(
            axes[1].plot(
                live_data.distanceTravelled[c], 
                live_data.metrePerSecond[c], 
                label=c
            )
        )
        lines[2].extend(
            axes[2].plot(
                live_data.distanceTravelled[c], 
                live_data.strokeRate[c], 
                label=c
            )
        )

    axes[2].set_ylim(25, 55)
    axes[2].set_xlim(0, 2000)
    axes[0].set_ylabel('PGMT')
    axes[1].set_ylabel('m/s')
    axes[2].set_ylabel('stroke rate')
    axes[2].set_xlabel('Distance')
    axes[0].legend(
        bbox_to_anchor=(0., 1.02, 1., .152), 
        loc='upper left', 
        ncol=len(countries),
        mode="expand", 
        borderaxespad=0.)

    return f, axes, lines