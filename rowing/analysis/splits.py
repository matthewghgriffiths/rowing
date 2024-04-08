
import os
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

from . import geodesy
from .utils import is_pareto_efficient, distance_to_km


_STANDARD_DISTANCES = {
    '250m': 0.25,
    '500m': 0.5,
    '1km': 1,
    '1.5km': 1.5,
    '2km': 2,
    '3km': 3,
    '5km': 5,
    '7km': 7,
    '10km': 10,
}

_file_path = Path(os.path.abspath(__file__))
_module_path = _file_path.parent
_DATA_PATH = _module_path.parent.parent / 'data'

_LOCATION_DATA = {
    'cam': _DATA_PATH / 'cam_locations.tsv',
    'ely': _DATA_PATH / 'ely_locations.tsv',
    'cav': _DATA_PATH / 'cav_locations.tsv',
    'tideway': _DATA_PATH / 'tideway_locations.tsv',
    'henley': _DATA_PATH / 'henley_locations.tsv',
    'dorney': _DATA_PATH / 'dorney_locations.tsv',
    'cerla': _DATA_PATH / 'Cerla_locations.tsv',
    'Shiplake': _DATA_PATH / 'Shiplake_locations.tsv',
    'Wycliffe': _DATA_PATH / 'Wycliffe_locations.tsv',
}


def load_place_locations(loc=None):
    if loc is None:
        loc = _LOCATION_DATA
    elif loc in _LOCATION_DATA:
        loc = [loc]

    return {
        p: pd.read_table(_LOCATION_DATA[p], index_col=0)
        for p in loc
    }


def load_landmarks(loc=None):
    return load_location_landmarks(loc).droplevel(0)


def load_location_landmarks(loc=None):
    return pd.concat(load_place_locations(loc), names=["location", "landmark"])


def find_all_crossing_times(positions, locations=None, thresh=0.5):
    locations = load_location_landmarks() if locations is None else locations

    positions = positions.reset_index(drop=True)
    names = list(locations.index.names) + ["distance"]

    times = pd.concat({
        loc: find_crossing_times(positions, pos, thresh=thresh)
        for loc, pos in locations.iterrows()
    },
        names=names
    ).sort_index(level=-1).reset_index()

    times['leg'] = 0
    # legs = times[names[:-1] + ['leg']]
    duplicates = times[names[:-1] + ['leg']].duplicated(keep='first')
    while duplicates.any():
        times['leg'] += np.maximum.accumulate(duplicates)
        duplicates = times[names[:-1] + ['leg']].duplicated(keep='first')

    return times.set_index(['leg'] + names)[0]


def find_all_crossing_data(positions, locations=None, thresh=0.5, cols=None):
    crossing_times = find_all_crossing_times(positions, locations, thresh)
    crossing_data = crossing_times.to_frame("time")

    if cols:
        crossings = pd.concat([
            crossing_times,
            pd.Series([pd.NaT], [("",) * crossing_times.index.nlevels])
        ]).index
        intervals = pd.IntervalIndex.from_breaks(
            crossing_times, closed='neither')
        groups = crossings[intervals.get_indexer(positions.time)]
        crossing_groups = positions.groupby(groups)
        for c in cols:
            crossing_data[c] = crossing_groups[c].mean().reindex(
                crossing_times.index)

    return crossing_data


def calc_timings(loc_times):
    times = loc_times.values
    loc_timings = pd.DataFrame(
        times[:, None] - times[None, :],
        index=loc_times.index,
        columns=loc_times.index
    )
    distances = np.array(loc_times.index.get_level_values('distance'), float)
    dist_diffs = 2 * (distances[:, None] - distances[None, :])
    dist_diffs[np.tril_indices(len(distances))] = 1

    loc_timings /= dist_diffs

    return pd.concat({'splits': pd.concat({'times': loc_timings})}, axis=1)


def get_location_timings(positions, locations=None, thresh=0.5):
    locations = load_landmarks() if locations is None else locations
    loc_times = find_all_crossing_times(
        positions, locations, thresh=thresh
    )
    times = loc_times.values
    loc_timings = pd.DataFrame(
        times[:, None] - times[None, :],
        index=loc_times.index,
        columns=loc_times.index
    )
    distances = np.array(loc_times.index.get_level_values('distance'), float)
    dist_diffs = 2 * (distances[:, None] - distances[None, :])
    dist_diffs[np.tril_indices(len(distances))] = 1

    loc_timings /= dist_diffs

    return pd.concat({'splits': pd.concat({'times': loc_timings})}, axis=1)


def find_crossing_times(positions, loc, thresh=0.5):
    close_points = geodesy.haversine_km(positions, loc) < thresh

    close_positions = positions[close_points].copy()
    close_positions.bearing = loc.bearing + 90

    intersections = pd.DataFrame.from_dict(
        geodesy.path_intersections(close_positions, loc)._asdict()
    )
    bearings = geodesy.bearing(intersections, loc)
    sgns = np.sign(np.cos(np.radians(bearings - loc.bearing)))
    if not sgns.size:
        return pd.Series([], dtype=float)

    crossings = bearings.index[sgns != sgns.shift(fill_value=sgns.iloc[0])]

    def weight(*ds):
        return ds[0] / sum(ds)

    crossing_weights = pd.Series([
        weight(*geodesy.haversine(intersections.loc[i:i+2], loc))
        for i in crossings
    ],
        index=crossings,
        dtype=float,
    )
    crossings1 = np.clip((crossings + 1), 0, positions.index[-1])
    time_deltas = (
        positions.time[crossings1].values - positions.time[crossings].values
    )
    crossing_times = (
        positions.time[crossings] + time_deltas * crossing_weights
    )
    distance_deltas = (
        positions.distance[crossings1].values
        - positions.distance[crossings].values
    )
    crossing_distances = (
        positions.distance[crossings] + distance_deltas * crossing_weights
    )
    crossing_times.index = crossing_distances.round(3)
    crossing_times.index.name = 'distance'

    return crossing_times


def get_distance_to_locations(activity_data, locs=None):
    if not isinstance(locs, pd.DataFrame):
        locs = pd.concat({
            l: pos.mean() for l, pos in load_place_locations(locs).items()
        }).unstack()

    loc_dists = locs.T.apply(
        lambda x: geodesy.haversine_km(activity_data, x)
    )
    return loc_dists, locs


def get_closest_locations(loc_dists, locs=None):
    if not isinstance(locs, pd.DataFrame):
        locs = pd.concat({
            l: pos.mean() for l, pos in load_place_locations(locs).items()
        }).unstack()

    return pd.DataFrame({
        'location': locs.index[loc_dists.values.argmin(1)],
        'distance': loc_dists.values.min(1),
    }, index=loc_dists.index)


def group_positions(positions, locations=None, freq='1h', thresh=10, update=False):
    if locations is None:
        locations = load_location_landmarks().groupby(level=0).mean()

    if not update:
        positions = positions.copy()
    location_distances = get_distance_to_locations(
        positions, locations
    )[0]
    closest_location = location_distances.idxmin(1)
    closest_distance = location_distances.min(1)
    closest_location[closest_distance > thresh] = np.nan

    positions['location'] = closest_location.reindex(
        positions.index
    )
    position_groups = positions.groupby(
        ["location", pd.Grouper(key="time", freq=freq)]
    ).size().index.to_frame(False)
    grouped_locations = locations.loc[
        position_groups["location"]
    ].reset_index()

    return pd.concat([position_groups, grouped_locations], axis=1)


def find_best_times(positions, distance, cols=None):
    positions = positions.reset_index()

    pos_distances = (
        positions.distance
        + np.minimum(positions.distance.diff(), 0).fillna(0)
    )

    total_distance = pos_distances.iloc[-1]
    time_elapsed = positions.timeElapsed.dt.total_seconds()
    sel = pos_distances + distance < total_distance
    distances = pos_distances[sel]
    end_distances = distances + distance

    dist_elapsed = np.interp(
        end_distances, pos_distances, time_elapsed
    )

    dist_times = dist_elapsed - time_elapsed[sel]
    best_ordering = dist_times.argsort().values

    best = []
    unblocked = np.ones_like(best_ordering, dtype=bool)
    n = unblocked.size
    while unblocked.any():
        next_best = best_ordering[unblocked[best_ordering]][0]
        best.append(next_best)
        i0 = end_distances.searchsorted(distances[next_best])
        i1 = distances.searchsorted(end_distances[next_best])
        unblocked[i0:i1] = False
        n1 = unblocked.sum()
        if n == n1:
            best = []
            break
        n = n1

    best_times = pd.to_timedelta(dist_times[best], 'S')
    best_timesplits = pd.DataFrame.from_dict({
        'time': best_times,
        'split': best_times / distance / 2
    })
    if cols:
        dist_cols = positions.set_index('distance')[cols]
        avg_col_vals = pd.DataFrame([
            dist_cols[d:d + distance][cols].mean(0)
            for d in distances[best]
        ], columns=cols).round(1)
        avg_col_vals.index = best
        best_timesplits = pd.concat([best_timesplits, avg_col_vals], axis=1)

    best_timesplits.index = distances[best].round(3)
    return best_timesplits


def find_all_best_times(positions, distances=None, cols=None):
    distances = distances or _STANDARD_DISTANCES
    if pd.Index(positions.distance).is_monotonic_increasing:
        return pd.concat({
            name: find_best_times(positions, distance, cols=cols)
            for name, distance in distances.items()
        },
            names=('length', 'distance'),
        )
    return pd.DataFrame([])


def process_activities(activities, locations=None, cols=None):
    activity_id = activities.index.names[0]
    each_activity = activities.groupby(level=0)
    activity_info = pd.DataFrame({
        "startTime": each_activity.time.min(),
        "totalDistance": each_activity.distance.max(),
    }).sort_values("startTime", ascending=False)
    best_times = each_activity.apply(
        lambda x: find_all_best_times(
            x.droplevel(0),
            cols=cols
        )
    ).join(activity_info).reset_index().sort_values(
        ["startTime", "length", "split"],
        key=lambda s: s.map(distance_to_km) if s.name == 'length' else s
    ).set_index(
        [activity_id, 'startTime', 'totalDistance', 'length', 'distance']
    )

    location_timings = (
        (actid, get_location_timings(activity.droplevel(0)))
        for actid, activity in each_activity if not activity.empty
    )
    location_timings = {
        actid: timings
        for actid, timings in location_timings if not timings.empty
    }
    return activity_info, best_times, location_timings


def calc_pareto_front(positions):
    i, j = np.triu_indices(len(positions), 1)

    time_diffs = (
        positions.timeElapsed[j].dt.total_seconds().values
        - positions.timeElapsed[i].dt.total_seconds().values
    )
    dist_diffs = (
        positions.distance[j].values - positions.distance[i].values)
    speeds = 1000 * np.nan_to_num(dist_diffs / time_diffs)
    mask = is_pareto_efficient(np.c_[-dist_diffs, -speeds])
    return pd.DataFrame.from_dict({
        'distance': dist_diffs[mask],
        'speeds': speeds[mask],
    })


def calc_time_above_hr(
    activities,
    key='activity_id',
    hrs=None,
    rescale=3600
):
    hrs = pd.RangeIndex(60, 200, 1)
    key = key or activities.index.names[0]
    return activities.reset_index().groupby(
        [key, 'heart_rate']
    ).timeDelta.sum().dt.total_seconds().unstack(
        fill_value=0
    ).sort_index(
        axis=1, ascending=False
    ).cumsum(1).reindex(
        hrs, axis=1
    ) / rescale


def get_piece_times(crossing_times, start_landmark, finish_landmark):
    name, leg, *loc, landmark, distance = crossing_times.index.names
    start_times = crossing_times.xs(
        start_landmark, level=landmark).droplevel(distance)
    finish_times = crossing_times.xs(
        finish_landmark, level=landmark).droplevel(distance)
    times = pd.concat({
        "Elapsed time": crossing_times - start_times,
        "Time left": finish_times - crossing_times,
        "Time": crossing_times,
    }, axis=1)
    valid_times = (
        times.notna().all(axis=1)
        & (times['Time left'].dt.total_seconds() >= 0)
        & (times["Elapsed time"].dt.total_seconds() >= 0)
    )
    if valid_times.any():
        piece_data = times[valid_times].reset_index(distance).unstack()
        avg_distance = piece_data.distance.mean().sort_values()
        col_order = avg_distance.index
        piece_data.index = pd.MultiIndex.from_frame(
            start_times.loc[
                piece_data.index
            ].rename("Start Time").reset_index()[
                ["Start Time", name, leg]
            ]
        )
        piece_data = piece_data.sort_index(level=0)
        piece_distances = (
            piece_data.distance
            - piece_data.distance[start_landmark].values[:, None]
        )[col_order]
        piece_time = piece_data['Elapsed time'][col_order]
        avg_split = (
            piece_time * 0.5 / piece_distances).fillna(pd.Timedelta(0))
        interval_split = (
            piece_time.diff(axis=1) * 0.5 / piece_distances.diff(axis=1)
        ).fillna(pd.Timedelta(0))

        piece_data = {
            "Elapsed Time": piece_time,
            "Distance Travelled": piece_distances,
            "Average Split": avg_split,
            "Interval Split": interval_split,
            "Timestamp": piece_data['Time'][col_order],
            "Total Distance": piece_data[distance][col_order],
        }

        return piece_data


def get_interval_averages(X, time, timestamps):
    time_intervals = pd.IntervalIndex.from_breaks(timestamps)
    group_intervals = pd.cut(time, time_intervals).replace(
        dict(zip(time_intervals, timestamps.index[1:]))
    )
    t = (time - time.min()).dt.total_seconds()

    dt = t.diff().values[:, None] * X.notna()
    Xdt = X * dt

    intervalT = dt.groupby(group_intervals).sum()
    intervalX = Xdt.groupby(group_intervals).sum() / intervalT
    avgX = (intervalX * intervalT).cumsum() / intervalT.cumsum()
    return (
        avgX.dropna(axis=0, how="all").dropna(axis=1),
        intervalX.dropna(axis=0, how="all").dropna(axis=1),
    )


def get_piece_gps_data(
    positions,
    piece_distances,
    piece_timestamps,
    start_landmark,
    finish_landmark,
    landmark_distances,
):
    positions = positions.reset_index(drop=True).loc[
        positions.distance.searchsorted(
            piece_distances[start_landmark]
        ) - 1:
        positions.distance.searchsorted(
            piece_distances[finish_landmark]
        ) + 1
    ].copy()
    positions['timeElapsed'] = (
        positions.time - piece_timestamps[start_landmark]
    )
    positions[
        'Distance Travelled'
    ] = np.interp(
        positions.distance, piece_distances, landmark_distances
    )

    return positions
