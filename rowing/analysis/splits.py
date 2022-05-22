
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

_DATA_PATH = (Path(__file__) / "../../data").resolve()

_LOCATION_DATA = {
    'cam': _DATA_PATH / 'cam_locations.tsv',
    'ely': _DATA_PATH / 'ely_locations.tsv',
    'cav': _DATA_PATH / 'cav_locations.tsv',
    'tideway': _DATA_PATH / 'tideway_locations.tsv',
}


def load_place_locations(loc=None):
    if loc is None:
        loc = _LOCATION_DATA
    elif loc in _LOCATION_DATA:
        loc = [loc]

    return {
        l: pd.read_table(_LOCATION_DATA[l], index_col=0)
        for l in loc
    }

def load_locations(loc=None):        
    return pd.concat(list(
        load_place_locations(loc).values()
    ))#.set_index('location', drop=True)


def find_all_crossing_times(positions, locations=None, thresh=0.15):
    locations = load_locations() if locations is None else locations

    times = pd.concat({
        loc: find_crossing_times(positions, pos, thresh=thresh)
        for loc, pos in locations.iterrows()
    },
        names = ['location', 'distance']
    ).sort_index(level=1).reset_index()
    times['leg'] = (times.location == times.location.shift()).cumsum()

    return times.set_index(['leg', 'location', 'distance'])[0]


def get_location_timings(positions, locations=None, thresh=0.15):
    locations = load_locations() if locations is None else locations

    loc_times = find_all_crossing_times(
        positions, locations, thresh=thresh
    )
    times = loc_times.values
    loc_timings = pd.DataFrame(
        times[:, None] - times[None, :],
        index=loc_times.index, 
        columns=loc_times.index
    )
    distances = loc_times.index.get_level_values('distance')
    dist_diffs = 2 * (distances.values[:, None] - distances.values[None, :])
    dist_diffs[np.tril_indices(len(distances))] = 1

    loc_timings /= dist_diffs

    return pd.concat({'splits': pd.concat({'times': loc_timings})}, axis=1)


def find_crossing_times(positions, loc, thresh=0.15):
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

    time_deltas = (
        positions.time[crossings + 1].values - positions.time[crossings].values
    )
    crossing_times = (
        positions.time[crossings] + time_deltas * crossing_weights 
    )
    distance_deltas = (
        positions.distance[crossings + 1].values 
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


def find_best_times(positions, distance, cols=None):
    total_distance = positions.distance.iloc[-1]
    time_elapsed = positions.timeElapsed.dt.total_seconds()
    sel = positions.distance + distance < total_distance
    distances = positions.distance[sel]
    end_distances = distances + distance

    dist_elapsed = np.interp(
        end_distances, positions.distance, time_elapsed    
    )

    dist_times = dist_elapsed - time_elapsed[sel] 
    best_ordering = dist_times.argsort().values

    best = []
    unblocked = np.ones_like(best_ordering, dtype=bool)
    while unblocked.any():
        next_best = best_ordering[unblocked[best_ordering]][0]
        best.append(next_best)
        i0 = end_distances.searchsorted(distances[next_best])
        i1 = distances.searchsorted(end_distances[next_best])
        unblocked[i0:i1] = False
    
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
    return pd.concat({
        name: find_best_times(positions, distance, cols=cols)
        for name, distance in distances.items()
    },
        names = ('length', 'distance'),
    )


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
    ).join(activity_info).reset_index().set_index(
        [activity_id, 'startTime', 'totalDistance', 'length', 'distance']
    ).sort_index(
        level=['startTime', "length"],
        key=lambda index: 
            index if isinstance(index, pd.DatetimeIndex) 
            else index.map(distance_to_km)
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
        positions.timeElapsed[j].dt.total_seconds().values - 
        positions.timeElapsed[i].dt.total_seconds().values
    )
    dist_diffs =(
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