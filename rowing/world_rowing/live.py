
import logging
from pathlib import Path
import json
import threading
import time
import copy

import numpy as np
import pandas as pd
from scipy import stats

from tqdm.auto import tqdm

from rowing.world_rowing import api, utils, fields
# from . import api, utils, fields
from .api import (
    get_worldrowing_data, get_race_results, get_worldrowing_record,
    find_world_best_time, INTERMEDIATE_FIELDS, get_most_recent_competition,
    get_races, get_events, get_world_best_times,
    get_live_race
)
from .utils import (
    extract_fields, format_yaxis_splits, make_flag_box, update_fill_betweenx,
    update_fill_between, read_times, format_totalseconds, cached_property,
    format_timedelta
)

logger = logging.getLogger("world_rowing.livetracker")

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


def parse_livetracker_lane_data(data, field, index):
    lanes = {
        lane['Lane']: lane
        for lane in data['config']['lanes']
        if lane.get(field)
    }
    if not lanes:
        return pd.DataFrame([])

    parsed = pd.concat(
        {
            lane['DisplayName']: pd.json_normalize(
                lane[field]).set_index(index)
            for lane in lanes.values()
        },
        axis=1,
        names=[fields.raceBoats, field]
    ).swaplevel(0, 1, 1).rename(
        columns=fields.renamer(field), level=0
    )
    return parsed.sort_index(axis=1)


def parse_livetracker_data(data):
    live_data = parse_livetracker_lane_data(data, 'live', 'id')
    if live_data.empty:
        return live_data
    
    for boat, speed in live_data[
        fields.live_raceBoatTracker_metrePerSecond
    ].items():
        live_data[fields.split, boat] = pd.to_timedelta(
            500 / speed, unit='s'
        )

    return live_data


def parse_intermediates_data(data):
    intermediates = parse_livetracker_lane_data(
        data, 'intermediates', 'distance.DisplayName'
    )
    if pd.api.types.is_object_dtype(intermediates.index):
        intermediates.index = intermediates.index.str.extract(
            "([0-9]+)")[0].astype(int)
    if fields.intermediates_ResultTime in intermediates:
        for c, times in intermediates[fields.intermediates_ResultTime].items():
            intermediates[(fields.intermediates_ResultTime, c)
                          ] = pd.to_timedelta(times)

    return intermediates


def parse_livetracker_info(data):
    lane_info = pd.concat([
        pd.json_normalize(
            {k: lane[k] for k in lane.keys() - {"live", "intermediates"}}
        )
        for lane in data['config']['lanes']
    ]).set_index("DisplayName").rename(columns=fields.renamer("lane"))
    lane_info[fields.lane_ResultTime] = utils.read_times(
        lane_info[fields.lane_ResultTime])
    race_distance = data['config']['plot']['totalLength']
    return lane_info, race_distance


def parse_livetracker(data):
    live_boat_data = parse_livetracker_data(data)
    if live_boat_data.empty:
        live_boat_data = None
    intermediates = parse_intermediates_data(data)
    lane_info, race_distance = parse_livetracker_info(data)
    return live_boat_data, intermediates, lane_info, race_distance


def load_livetracker(race_id, cached=True):
    data = get_worldrowing_data("livetracker", race_id, cached=cached)
    return parse_livetracker(data)


def shift_down(s, shift, min_val=0):
    sp = (s - shift).clip(min_val, None)
    nzidx = (sp > min_val).idxmax()
    s0 = s.loc[:nzidx]
    sp.loc[:nzidx] = min_val + sp.loc[nzidx] * (s0 - s0.min()) / s0.max()
    return sp


def estimate_livetracker_times(live_boat_data, intermediates, lane_info, race_distance):
    intermediate_times = intermediates[
        fields.intermediates_ResultTime].apply(lambda s: s.dt.total_seconds())
    intermediate_distances = intermediate_times.index.values

    live_data = live_boat_data.loc[
        live_boat_data[fields.live_trackCount].min(1).sort_values().index
    ].reset_index(drop=True)
    countries = live_data.columns.levels[1]
    distances = live_data[fields.live_raceBoatTracker_distanceTravelled]
    speed = live_data[fields.live_raceBoatTracker_metrePerSecond]

    # Estimate times
    diffs = - distances.diff(-1).replace(0, np.nan)
    boat_time_diff = diffs / speed.replace(0, np.nan)
    mean_time_diff = boat_time_diff.mean(1).fillna(0)
    times = mean_time_diff.cumsum().rename(fields.live_time)

    # Make sure timepoints around intermediates are correct
    int_times = pd.Series(np.nan, live_data.index)
    for d in intermediate_distances:
        # timepoints before boat passes
        m0 = (
            (distances <= d) & (distances.shift(-1) > d)
        )
        d_times = (
            intermediate_times.loc[d] - (
                # backtrack time to time point if not at intermediate distance
                (d - distances) / speed * m0
            )[m0.any(axis=1)].max()
        ).groupby(m0.idxmax()).mean()
        int_times.update(d_times)

    start_err = (times - int_times).mean()
    times = shift_down(times, start_err)
    times -= start_err

    t = (int_times - times).dropna()
    times += np.interp(times.index, t.index, t)
    times = shift_down(times, 0)

    live_time_data = live_data.copy()
    live_time_data.index = times

    for c in countries:
        c_data = live_time_data.xs(c, axis=1, level=1)
        live_time_data[(fields.avg_speed, c)] = (
            c_data[fields.live_raceBoatTracker_distanceTravelled] / c_data.index
        )
        live_time_data[(fields.split, c)] = pd.to_timedelta(
            500 / live_time_data[fields.live_raceBoatTracker_metrePerSecond][c], unit='s', errors='coerce')
        live_time_data[(fields.avg_split, c)] = pd.to_timedelta(
            500. / live_time_data[fields.avg_speed, c], unit='s', errors='coerce')

    live_data = live_time_data.stack(1).reset_index().join(
        lane_info[[
            fields.lane_Rank,
            fields.lane_Lane, 
            fields.lane_ResultTime,
            fields.lane_country_CountryCode,
            fields.lane_country,
            fields.lane__finished,
            fields.lane_InvalidMarkResult,
            fields.lane_Remark,
        ]], 
        on=fields.raceBoats,
        rsuffix='_1'
    )
    live_data[fields.race_distance] = race_distance
    return live_data, intermediate_times


def get_races_livetracks(race_ids, max_workers=10, load_livetracker=load_livetracker, **kwargs):
    race_livetracks, errors = utils.map_concurrent(
        load_livetracker,
        {race_id: race_id for race_id in race_ids},
        singleton=True, max_workers=max_workers, **kwargs
    )
    logger.info("load_livetracker errors: %s", errors)
    results, errors = utils.map_concurrent(
        estimate_livetracker_times,
        race_livetracks, max_workers=max_workers, **kwargs
    )
    logger.info("estimate_livetracker_times errors: %s", errors)
    intermediates = pd.concat(
        {race_id: inters for race_id, (_, inters) in results.items()},
        axis=1
    )
    races_live_data = pd.concat(
        {race_id: live_data for race_id, (live_data, _) in results.items()},
        axis=0
    ).reset_index(drop=True)
    lane_info = pd.concat(
        {
            race_id: data[2] for race_id, data in race_livetracks.items()
        }, 
        axis=0
    ).reset_index(drop=True)
    return races_live_data, intermediates, lane_info


class RealTimeLivetracker:
    def __init__(
            self, race_id, realtime_sleep=3,
            replay=False, replay_index=0, replay_step=1
    ):
        self.race_id = race_id

        self.realtime_sleep = realtime_sleep
        self.livetracker_max_age = time.time()

        self.realtime_history = {}
        self.livetracker_history = {}
        self.replay = replay
        self.replay_data = None
        self.replay_index = replay_index
        self.replay_step = replay_step
        self._shutdown = False

    @classmethod
    def from_live_race(cls, **kwargs):
        race = api.get_live_race()
        if race is not None:
            return cls(race.name, **kwargs)

        logger.warning("RealTimeLivetracker: no live race could be found")

    def get_realtime(self):
        curr_time = time.time()
        logger.debug("RealTimeLivetracker.get_realtime")
        data = api.get_worldrowing_data(
            "livetracker", "live", self.race_id, cached=False
        )
        self.realtime_history[curr_time] = data
        return data

    def get_replay_realtime(self):
        if self.replay_data is None:
            self.replay_data = api.get_worldrowing_data(
                "livetracker", self.race_id)

        i = self.replay_index
        live_data = copy.deepcopy(self.replay_data)
        for lane, lane_data in enumerate(live_data['config']['lanes']):
            lane_count = len(lane_data['live'])

            if i >= lane_count - 1:
                s = slice(lane_count - 1, lane_count)
                lane_data["_finished"] = True
            else:
                s = slice(i, i+1)
                lane_data["_finished"] = False

            lane_data['live'] = lane_data['live'][s]
            live_data['config']['lanes'][lane] = lane_data

        self.replay_index += self.replay_step
        return live_data

    def get_livetracker(self):
        curr_time = time.time()
        logger.debug("RealTimeLivetracker.get_livetracker")
        self.r = r = api.request_worldrowing("livetracker", self.race_id)
        max_age = int(r.headers['Cache-Control'].split("=")[1])
        age = int(r.headers.get('Age', 0))
        logger.info(
            "livetracker max-age=%d age=%d", max_age, age
        )
        self.livetracker_age = curr_time - age
        self.livetracker_max_age = max_age + self.livetracker_age

        data = r.json()['data']
        self.livetracker_history[curr_time] = data
        return data

    def wait_for_livetrack(self):
        max_age = self.livetracker_max_age
        data = None
        while self.livetracker_age < max_age:
            wait = max_age - time.time()
            logger.info(
                "RealTimeLivetracker.wait_for_livetrack waiting=%.1fs", wait
            )
            time.sleep(wait)
            data = self.get_livetracker()

        return data

    def run(self):
        last_time = time.time()
        finished = False
        i = 0
        while not (finished or self._shutdown):
            logger.debug("RealTimeLivetracker.run %d", i)
            i += 1
            curr_time = time.time()
            if self.replay:
                data = self.get_replay_realtime()
            elif curr_time > self.livetracker_max_age:
                data = self.get_livetracker()
            else:
                data = self.get_realtime()

            finished = self.is_finished(data)
            yield data,

            if finished:
                logger.info("finished polling realtime data")
                return

            curr_time = time.time()
            sleep_time = max(0, (last_time + self.realtime_sleep) - curr_time)
            logger.debug(
                "RealTimeLivetracker.poll sleeping: "
                "realtime=%.2f, livetracker=%.2f",
                sleep_time, self.livetracker_max_age - curr_time
            )
            last_time = curr_time
            time.sleep(sleep_time)

    def is_finished(self, data):
        return all(lane['_finished'] for lane in data['config']['lanes'])


def update_dataframe(current, update, overwrite=False):
    current = current.reindex(update.index.union(current.index))
    current.update(update, overwrite=False)
    return current


class LiveRaceData:
    def __init__(self, race_id, **kwargs):
        self.race_id = race_id
        self.tracker = RealTimeLivetracker(race_id, **kwargs)

        self.livetracker = None
        self.intermediates = None
        self.lane_info = None
        self.race_distance = None
        self.new_points = None
        self.mutex = threading.Lock()

    @property 
    def distance(self):
        if self.livetracker is not None:
            return self.livetracker[fields.live_distanceOfLeader].max(1).max()
        return 0

    def update(self, data):
        (
            livetracker_update,
            inter_update,
            lane_update,
            self.race_distance
        ) = parse_livetracker(data)

        current_points = pd.Index([])
        with self.mutex:
            if self.livetracker is not None:
                current_points = self.livetracker.index
                livetracker_update = update_dataframe(
                    self.livetracker, livetracker_update)
            if self.intermediates is not None:
                inter_update = update_dataframe(
                    self.intermediates, inter_update)
                inter_update.index.name = fields.Distance

            if self.lane_info is not None:
                lane_update = update_dataframe(self.lane_info, lane_update)
            
            self.lane_info = lane_update 
            self.intermediates = inter_update
            if livetracker_update is not None:
                track_count = livetracker_update[fields.live_trackCount].mean(
                    axis=1).sort_values()
                self.livetracker = livetracker_update.loc[track_count.index]
                self.new_points = livetracker_update.index.difference(
                    current_points)

        return self

    def plot_data(self, facets=None):
        if self.livetracker is None:
            return None

        index_names = [
            fields.live_raceBoatTracker_id,
            fields.raceBoats,
            fields.live_raceBoatTracker_distanceTravelled
        ]
        stacked = self.livetracker.stack(
            1
        ).droplevel(0).reset_index().set_index(
            index_names
        )
        if facets:
            plot_data = stacked[
                facets
            ].reset_index().melt(
                index_names
            ).join(stacked[facets], on=index_names)
        else:
            plot_data = stacked.reset_index().melt(index_names)

        plot_data = fields.to_plotly_dataframe(plot_data.dropna(subset=["value"]))

        plot_data[fields.split] = pd.to_datetime(plot_data[fields.split])
        
        distance = plot_data[fields.live_distanceTravelled]
        filter_distance = min(distance.quantile(0.2), 100)
        
        data_filter = (
            (distance > filter_distance)
            & (distance < self.race_distance)
        )
        if data_filter.sum() < 10:
            data_filter = slice(None)

        facet_groups = plot_data[data_filter].groupby("live")
        facet_types = facet_groups.value.first().map(type)
        if facets is None:
            facets = facet_types.index[facet_types != str]

        facet_format = {f: True for f in facets}
        # facet_format['kilometrePersSecond'] = False
        facet_format[fields.split] = "|%-M:%S.%L"
            
        facet_max = facet_groups.value.max().reindex(facets)
        facet_min = facet_groups.value.min().reindex(facets)
        facet_ptp = facet_max - facet_min
        facet_data = pd.concat(
            [facet_min - facet_ptp*0.1, facet_max + facet_ptp*0.1, ],
            axis=1).apply(tuple, axis=1).rename("range").to_frame()
        facet_data['matches'] = None
        facet_data['title_text'] = facet_data.index
        facet_axes = facet_data.T.to_dict()

        facet_axes.setdefault(fields.split, {})['tickformat'] = "%-M:%S"

        for col in [
            fields.live_raceBoatTracker_distanceFromLeader,
            fields.split,
        ]:
            facet_axes[col]['range'] = facet_axes[col]['range'][::-1]

        return plot_data, facet_axes, facet_format
