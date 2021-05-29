
import numpy as np
import pandas as pd
from pandas.core.construction import extract_array

from .api import get_worldrowing_data, find_world_best_time, INTERMEDIATE_FIELDS
from .utils import extract_fields

RESULTS_FIELDS = {
    'id': ('id',),
    'boatId': ('boatId',),
    'countryId': ('countryId',),
    'worldBestTimeId': ('worldBestTimeId',),
    'raceId': ('raceId',),
    'DisplayName': ('DisplayName',),
    'Rank': ('Rank',),
    'Lane': ('Lane',),
    'WorldCupPoints': ('WorldCupPoints',),
    'InvalidMarkResult': ('InvalidMarkResult',),
    'Remark': ('Remark',),
    'ResultTime': ('ResultTime',),
    # 'raceBoatIntermediates': ('raceBoatIntermediates',),
}

# def get_race_livetracker(race_id, gmt=None, cached=True, race_distance=2000):
#     data = get_worldrowing_data('livetracker', race_id, cached=cached)
#     has_livedata = data and data['live']
#     if not has_livedata:
#         return pd.DataFrame([])

#     results = pd.DataFrame.from_records(
#         extract_fields(result, RESULTS_FIELDS)
#         for result in data['intermediates'] if result['ResultTime']
#     )
#     results.ResultTime = pd.to_timedelta(results.ResultTime)
#     intermediates = pd.DataFrame.from_records(
#         extract_fields(inter, INTERMEDIATE_FIELDS)
#         for result in data['intermediates'] if result['ResultTime']
#         for inter in sorted(
#             result['raceBoatIntermediates'],
#             key=lambda x: x['ResultTime']
#         )
#     )
#     intermediates.ResultTime = pd.to_timedelta(intermediates.ResultTime)

#     gmt = gmt or find_world_best_time(
#         race_id=race_id
#     ).ResultTime.total_seconds()

#     lane_boat = {
#         lane['Lane']: lane for lane in data['config']['lanes']
#     }
#     rank_boat = {
#         lane['Rank']: lane for lane in data['config']['lanes']
#     }
#     lane_cnt = {r: lane['DisplayName'] for r, lane in lane_boat.items()}
#     rank_cnt = {r: lane['DisplayName'] for r, lane in rank_boat.items()}
#     countries = [lane_cnt[i] for i in sorted(lane_cnt)]
    
#     live_boat_data = {
#         'currentPosition': {},
#         'distanceTravelled': {},
#         'distanceFromLeader': {},
#         'strokeRate': {},
#         'metrePerSecond': {},
#     }
#     for cnt in countries:
#         for live_data in live_boat_data.values():
#             live_data[cnt] = []

#     for live_data in data['live']:
#         for tracker in live_data['raceBoatTrackers']:
#             cnt = lane_cnt[tracker['startPosition']]
#             for key, live_data in live_boat_data.items():
#                 live_data[cnt].append(tracker[key])

#     maxlen = max(
#         max(map(len, live_data.values()))
#         for live_data in live_boat_data.values()
#     )
#     for key, live_data in live_boat_data.items():
#         for cnt, cnt_data in list(live_data.items()):
#             cnt_len = len(cnt_data)
#             if cnt_len == 0:
#                 del live_data[cnt]
#             elif cnt_len < maxlen:
#                 cnt_data.extend(cnt_data[-1:] * (maxlen - cnt_len))

#     live_boat_data = pd.concat(
#         {
#             key: pd.DataFrame.from_dict(live_data) 
#             for key, live_data in live_boat_data.items()
#         },
#         axis=1
#     )

#     n_countries = len(live_boat_data.distanceTravelled.columns)
#     # Estimate times for each distance
#     boat_times = np.diff(
#         np.c_[
#             np.zeros(n_countries), 
#             live_boat_data.distanceTravelled.values.T
#         ], 
#         axis=1
#     ).T / live_boat_data.metrePerSecond
#     for col in boat_times:
#         live_boat_data['time', col] = boat_times[col].cumsum()

    
#     gmt_speed = race_distance / gmt
#     countries = live_boat_data.time.columns
#     for cnt in countries:
#         live_boat_data['GMT', cnt] = \
#             live_boat_data.distanceTravelled[cnt]/gmt_speed
#     for cnt in countries:
#         live_boat_data['PGMT', cnt] = \
#             live_boat_data.GMT[cnt] / live_boat_data.time[cnt]


#     return live_boat_data, results, intermediates

def get_race_livetracker(race_id, gmt=None, cached=True, race_distance=2000):
    data = get_worldrowing_data('livetracker', race_id, cached=cached)
    if data and data['live']:
        live_data = parse_livetracker_data(data)
        gmt = gmt or find_world_best_time(
            race_id=race_id
        ).ResultTime.total_seconds()
        live_data = calculate_pgmts(
            live_data, gmt=gmt, 
            race_distance=race_distance
            )
    else:
        live_data = pd.DataFrame([])
    
    if data and data['intermediates']:
        results, intermediates = parse_livetracker_results(data)
    else:
        results = pd.DataFrame([])
        intermediates = pd.DataFrame([])

    return live_data, results, intermediates


def parse_livetracker_results(data):
    results = pd.DataFrame.from_records(
        extract_fields(result, RESULTS_FIELDS)
        for result in data['intermediates'] if result['ResultTime']
    )
    intermediates = pd.DataFrame.from_records(
        extract_fields(inter, INTERMEDIATE_FIELDS)
        for result in data['intermediates'] if result['ResultTime']
        for inter in sorted(
            result['raceBoatIntermediates'],
            key=lambda x: x['ResultTime']
        )
    )
    if 'ResultTime' in results.columns:
        results.ResultTime = pd.to_timedelta(results.ResultTime)
    if 'ResultTime' in intermediates.columns:
        intermediates.ResultTime = pd.to_timedelta(intermediates.ResultTime)
    return results, intermediates


def parse_livetracker_data(data):
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
    return live_boat_data


def calculate_pgmts(live_boat_data, gmt, race_distance=2000):
    n_countries = len(live_boat_data.distanceTravelled.columns)
    distances = np.c_[
        np.zeros(n_countries), 
        live_boat_data.distanceTravelled.values.T
    ].T
    boat_diffs = np.diff(distances, axis=0)
    boat_times = boat_diffs / live_boat_data.metrePerSecond
    live_boat_data['time'] = np.ma.masked_array(
        boat_times, mask = boat_diffs==0
        ).mean(1).data.cumsum()
    
    gmt_speed = race_distance / gmt
    countries = live_boat_data.distanceTravelled.columns
    for cnt in countries:
        live_boat_data['GMT', cnt] = \
            live_boat_data.distanceTravelled[cnt]/gmt_speed
    for cnt in countries:
        live_boat_data['PGMT', cnt] = \
            live_boat_data.GMT[cnt] / live_boat_data.time

    return live_boat_data


def estimate_intermediate_times1(live_data):
    distances = [500, 1000, 1500, 2000]
    clips = (
        (c, live_data.distanceTravelled[c].searchsorted(2000) + 1)
        for c in live_data.distanceTravelled.columns
    )
    return pd.concat({
        country: pd.Series(
            np.interp(
                distances, 
                live_data.distanceTravelled[country][:i], 
                live_data.time[country][:i]
            ), 
            index=distances
        )
        for country, i in clips
    }).unstack().sort_index()


def estimate_intermediate_times(live_data):
    distances = [500, 1000, 1500, 2000]
    clips = (
        (c, live_data.distanceTravelled[c].searchsorted(2000) + 1)
        for c in live_data.distanceTravelled.columns
    )
    return pd.concat({
        country: pd.Series(
            np.interp(
                distances, 
                live_data.distanceTravelled[country][:i], 
                live_data.time[:i]
            ), 
            index=distances
        )
        for country, i in clips
    }).unstack().sort_index()


def extract_intermediate_times(results, intermediates):
    true_intermediates = pd.merge(
        intermediates, results,
        left_on='raceBoatId', 
        right_on='id',
        suffixes=(None, '_r')
    )[['DisplayName', 'distance', 'ResultTime']].set_index(
        ['DisplayName', 'distance']
    ).ResultTime.dt.total_seconds().unstack()[
        ['d500m', 'd1000m', 'd1500m', 'd2000m']
    ].sort_index()
    true_intermediates.columns = [500, 1000, 1500, 2000]
    return true_intermediates

    
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