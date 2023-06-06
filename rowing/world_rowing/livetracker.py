
import logging
from pathlib import Path 
import json 

import numpy as np
import pandas as pd
from scipy import stats

from tqdm.auto import tqdm 

from . import api, utils 
from .api import (
    get_worldrowing_data, get_race_results, get_worldrowing_record,
    find_world_best_time, INTERMEDIATE_FIELDS, get_most_recent_competition,
    get_competition_races, get_competition_events, get_world_best_times, 
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


class RaceTracker:
    def __init__(
            self, race_id,
            gmt=None,
            colors=None,
            live_data=None,
            results=None,
            intermediates=None,
            race_distance=None,
            intermediate_distances=None,
            live=False, 
    ):
        self.race_id = race_id
        self.live = live

        self.gmt = gmt or find_world_best_time(
            race_id=race_id
        ).ResultTime.total_seconds()

        self._colors = colors

        self.live_data = live_data
        self.results = results
        self.intermediates = intermediates
        self.race_distance = race_distance
        self.completed = False
        self.intermediate_distances = intermediate_distances or [500, 1000, 1500, 2000]

    @classmethod
    def load_live_race(cls, fisa=True, competition=None, **kwargs):
        live_race = get_live_race(fisa=fisa, competition=competition)
        if live_race is not None:
            kwargs.setdefault('live', True)
            return cls(live_race.name, **kwargs)

    @property
    def final_results(self):
        if self.results is not None:
            final_results = self.results.set_index(
                'DisplayName'
            ).ResultTime.dt.total_seconds().sort_values()
            final_results.index.name = 'country'
            return final_results

    @property
    def colors(self):
        if self._colors:
            return self._colors
        else:
            import matplotlib.pyplot as plt
            return plt.rcParams['axes.prop_cycle'].by_key()['color']

    @property
    def intermediate_results(self):
        if self.intermediates is None:
            return None
        elif len(self.intermediates):
            return get_intermediate_times(self.intermediates)
        else:
            return self.intermediates

    @cached_property
    def race_details(self):
        return get_worldrowing_record('race', self.race_id)

    @cached_property
    def race_boats(self):
        return get_race_results(race_id=self.race_id).reset_index()

    @cached_property
    def countries(self):
        return self.race_boats.Country

    @cached_property
    def lane_country(self):
        return self.race_boats.set_index('Lane').Country.sort_index()

    @cached_property
    def country_lane(self):
        return self.race_boats.set_index('Country').Lane.sort_values()

    @property
    def country_colors(self):
        colors = pd.Series(
            dict(zip(self.countries, self.colors)),
            name='color'
        )
        colors.index.name = 'country'
        return colors

    @cached_property
    def event_id(self):
        return self.race_details.eventId

    @cached_property
    def event_details(self):
        return get_worldrowing_record('event', self.event_id)

    @cached_property
    def competition_id(self):
        return self.event_details.competitionId

    @cached_property
    def competition_details(self):
        return get_worldrowing_record(
            'competition', self.competition_id)

    def update_livedata(self):
        self.live_data, self.results, self.intermediates, self.race_distance = \
            get_race_livetracker(
                self.race_id,
                gmt=self.gmt,
                cached=False,
                live=self.live, 
            )

        if self.race_distance in self.intermediate_results.index:
            if self.intermediate_results.loc[self.race_distance].notna().all():
                self.completed = True

        return self.live_data, self.intermediate_results

    def stream_livedata(self):
        live = self.live
        while not self.completed:
            self.live = True
            yield self.update_livedata()

        self.live = live

    def _by_country(self, values):
        try:
            return {cnt: values[cnt] for cnt in self.countries}
        except (TypeError, IndexError):
            return {cnt: values for cnt in self.countries}

    def bar(self, heights, bottom=None, yerr=None, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        bars = ax.bar(
            self.lane_country.index,
            heights[self.lane_country],
            bottom=bottom,
            yerr=yerr,
            tick_label=self.lane_country,
            color=self.country_colors[self.lane_country],
            **kwargs
        )
        return dict(zip(self.lane_country, bars))

    def update_bar(self, bars, heights, bottom=None):
        if bottom:
            bottoms = self._by_country(bottom)

        for cnt, height in heights.items():
            bar = bars[cnt]
            bar.set_height(height)
            if bottom:
                x, _ = bar.get_xy()
                bar.set_xy((x, bottoms[cnt]))

    def get_bar_lims(self, bars):
        bar_dims = [
            b.get_xy() + (b.get_width(), b.get_height(), b)
            for b in bars.values()
        ]
        x, y = np.hstack([
            [[x, x + w], [y, y + h]]
            for x, y, w, h, b in bar_dims
        ])
        return (x.min(), x.max()), (y.min(), y.max())

    def plot_flags(
            self, *args, ax=None,
            zoom=0.04,
            box_alignment=(0.5, 0.),
            **kwargs
    ):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        if len(args) == 1:
            y, = args
            x = self.country_lane[y.index]
        else:
            x, y = args

        flags = {}
        for cnt in y.index:
            xy = x[cnt], y[cnt]
            try:
                flag = make_flag_box(
                    cnt[:3],
                    xy,
                    zoom=zoom,
                    box_alignment=box_alignment,
                    **kwargs
                )
                ax.add_artist(flag)
                flags[cnt] = flag
            except KeyError:
                logger.warning("could not find flag for %s", cnt)


        return flags

    def update_flags(
        self, flags, *args
    ):
        if len(args) == 1:
            y, = args
            x = self.country_lane[y.index]
        else:
            x, y = args

        for cnt in flags:
            flags[cnt].xybox = flags[cnt].xy = x[cnt], y[cnt]

    def violin(
            self, y_dens, ax=None, width=0.8, alpha=0.6, outline=True,
            set_xticks=True, outline_kws=None, **kwargs
    ):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        violins = {}
        lines = {}
        if width:
            y_dens = y_dens / y_dens.max(0) * width / 2

        for x0, cnt in self.lane_country.items():
            if cnt not in y_dens:
                continue

            violins[cnt] = ax.fill_betweenx(
                y_dens.index,
                -y_dens[cnt] + x0,
                y_dens[cnt] + x0,
                color=self.country_colors[cnt],
                label=cnt,
                alpha=alpha,
                **kwargs
            )
            if outline:
                lines[cnt], = ax.plot(
                    np.r_[-y_dens[cnt] + x0, y_dens[cnt].iloc[::-1] + x0],
                    np.r_[y_dens.index, y_dens.index[::-1]],
                    color=self.country_colors[cnt],
                    label=cnt,
                    **(outline_kws or {})
                )

        if set_xticks:
            ax.set_xticks(self.lane_country.index)
            ax.set_xticklabels(self.lane_country)
        return violins, lines

    def update_violins(
            self, violins, lines, y_dens, width=0.8
    ):
        if width:
            y_dens = y_dens / y_dens.max(0) * width / 2

        for x0, cnt in self.lane_country.items():
            if cnt not in y_dens:
                continue

            update_fill_betweenx(
                violins[cnt],
                y_dens.index,
                -y_dens[cnt] + x0,
                y_dens[cnt] + x0,
            )
            line = lines.get(cnt, None)
            if line:
                line.set_data(
                    np.r_[-y_dens[cnt] + x0, y_dens[cnt].iloc[::-1] + x0],
                    np.r_[y_dens.index, y_dens.index[::-1]],
                )

    def calc_finish_densities(self, pred_finish, finish_std, dy_range=None):
        finish_std = self._by_country(finish_std)
        dy_range = (
            dy_range or 4 * max(std for _, std in finish_std.items())
        )
        ylim = (pred_finish.min() - dy_range, pred_finish.max() + dy_range)
        y = np.linspace(*ylim, 1000)
        y_dens = pd.DataFrame({
            cnt: stats.norm(
                loc=pred_finish[cnt],
                scale=finish_std[cnt]
            ).pdf(y)
            for cnt in self.lane_country
        },
            index=y
        )
        return y_dens, ylim

    def plot_finish(
            self, pred_finish, finish_std, dy_range=None,
            ax=None, set_lims=True, **kwargs
    ):
        import matplotlib.pyplot as plt
        y_dens, ylim = self.calc_finish_densities(
            pred_finish, finish_std, dy_range=dy_range
        )
        violins, lines = self.violin(y_dens, ax=ax, **kwargs)
        if set_lims:
            ax = ax or plt.gca()
            ax.set_ylim(*ylim)

        return violins, lines

    def update_plot_finish(
            self,
            violins, lines,
            pred_finish, finish_std, dy_range=None, width=0.8,
    ):
        y_dens, ylim = self.calc_finish_densities(
            pred_finish, finish_std, dy_range=dy_range
        )
        self.update_violins(
            violins, lines, y_dens, width=width,
        )
        return ylim

    def plot(
            self, *args, ax=None, maxdistance=None,
            set_lims=True,
            **kwargs
    ):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        lines = {}
        if len(args) == 1:
            y, = args
            x = self._by_country(y.index)
        else:
            x, y = args
            x = self._by_country(x)

        for cnt in y.columns:
            lines[cnt], = ax.plot(
                x[cnt],
                y[cnt],
                label=cnt,
                color=self.country_colors[cnt],
                **kwargs
            )
        if set_lims:
            ax.set_xlim(0, maxdistance or 2000)
        # ax.set_xlabel('distance (m)')

        return lines

    def update_plot(self, lines, x, y):
        x = self._by_country(x)
        for cnt, c_y in y.items():
            lines[cnt].set_data(
                x[cnt], c_y
            )

    def plot_uncertainty(
            self, *args, ax=None, alpha=0.2,
            maxdistance=None, fill_kws=None, **kwargs
    ):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        lines = {}
        collections = {}
        if len(args) == 2:
            y, yerr = args
            distance = y.index
        else:
            distance, y, yerr = args

        distance = self._by_country(distance)
        # if covariance passed extract square root of diagonal
        yerr = {
            cnt: np.diag(err)**0.5 if np.ndim(err) == 2 else err
            for cnt, err in self._by_country(yerr).items()
        }

        for cnt in y.columns:
            lines[cnt], = ax.plot(
                distance[cnt],
                y[cnt],
                label=cnt,
                color=self.country_colors[cnt],
                **kwargs
            )
            collections[cnt] = ax.fill_between(
                distance[cnt],
                y[cnt] - yerr[cnt],
                y[cnt] + yerr[cnt],
                label=cnt,
                color=self.country_colors[cnt],
                alpha=alpha,
                **(fill_kws or {})
            )
        ax.set_xlim(0, maxdistance or 2000)
        ax.set_xlabel('distance / m')

        return lines, collections

    def update_plot_uncertainty(
            self, lines, collections,
            *args,
    ):
        if len(args) == 2:
            y, yerr = args
            distance = y.index
        else:
            distance, y, yerr = args

        distance = self._by_country(distance)
        # if covariance passed extract square root of diagonal
        yerr = {
            cnt: np.diag(err)**0.5 if np.ndim(err) == 2 else err
            for cnt, err in self._by_country(yerr).items()
        }
        for cnt in y.columns:
            lines[cnt].set_data(
                distance[cnt],
                y[cnt],
            )
            update_fill_between(
                collections[cnt],
                distance[cnt],
                y[cnt] - yerr[cnt],
                y[cnt] + yerr[cnt],
            )

    def plot_pace(self, ax=None, **kwargs):
        distance = self.live_data.distanceTravelled
        pace = 500 / self.live_data.metrePerSecond
        ax, lines = self.plot(distance, pace, ax=ax, **kwargs)
        format_yaxis_splits(ax)
        ax.set_ylabel('pace / 500m')

        return ax, lines

    def plot_speed(self, ax=None, **kwargs):
        distance = self.live_data.distanceTravelled
        speed = self.live_data.metrePerSecond
        ax, lines = self.plot(distance, speed, ax=ax, **kwargs)
        ax.set_ylabel('speed / m/s')

        return ax, lines

    def plot_distance_from_leader(self, ax=None, **kwargs):
        distance = self.live_data.distanceTravelled
        speed = self.live_data.distanceFromLeader
        ax, lines = self.plot(distance, speed, ax=ax, **kwargs)
        ax.set_ylabel('distanceFromLeader / m')
        return ax, lines


def shift_down(s, shift, min_val=0):
    sp = (s - shift).clip(min_val, None)
    nzidx = (sp > min_val).idxmax()
    s0 = s.loc[:nzidx]
    sp.loc[:nzidx] = min_val + sp.loc[nzidx] * (s0 - s0.min()) / s0.max()
    return sp


def load_livetracker(race_id, cached=True):
    data = get_worldrowing_data("livetracker", race_id, cached=cached)
    live_boat_data = parse_livetracker_data(data)
    intermediates = parse_intermediates_data(data)
    lane_info = pd.concat([
        pd.json_normalize(
            {k: lane[k] for k in lane.keys() - {"live", "intermediates"}}
        )
        for lane in data['config']['lanes']
    ]).set_index("DisplayName")
    lane_info['ResultTime'] = utils.read_times(lane_info.ResultTime)
    race_distance = data['config']['plot']['totalLength']
    return live_boat_data, intermediates, lane_info, race_distance


def estimate_livetracker_times(live_boat_data, intermediates, lane_info, race_distance):
    intermediate_times = intermediates.ResultTime.apply(lambda s: s.dt.total_seconds())
    intermediate_distances = intermediates.distance.values[:, 0].astype(float)
    intermediate_times.index = intermediate_distances

    live_data = live_boat_data.loc[
        live_boat_data.trackCount.min(1).sort_values().index
    ].reset_index(drop=True)

    countries = live_data.columns.levels[1]
    distances = live_data.distanceTravelled
    speed = live_data.metrePerSecond

    # Estimate times
    diffs = - distances.diff(-1).replace(0, np.nan)
    boat_time_diff = diffs / speed.replace(0, np.nan)
    mean_time_diff = boat_time_diff.mean(1).fillna(0)
    times = mean_time_diff.cumsum().rename("time")

    # Make sure timepoints around intermediates are correct
    int_times = pd.Series(np.nan, live_data.index)
    for d in intermediate_distances:
        #timepoints before boat passes 
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
        live_time_data[("avg_speed", c)] = (
            c_data.distanceTravelled / c_data.index
        )

    live_data = live_time_data.stack(1).reset_index().join(
        lane_info[[
            'Rank', 'ResultTime', 
            'country.CountryCode', 'country.DisplayName',
            '_finished', 'InvalidMarkResult', "Remark"
        ]], on='boat'
    )
    live_data['raceDistance'] = race_distance
    return live_data, intermediate_times 



def get_races_livetracks(race_ids, max_workers=10, load_livetracker=load_livetracker, **kwargs):
    race_livetracks, errors = utils.map_concurrent(
        load_livetracker, 
        {race_id: race_id for race_id in race_ids},
        singleton=True, max_workers=max_workers, **kwargs
    )
    results, errors = utils.map_concurrent(
        estimate_livetracker_times, 
        race_livetracks, max_workers=max_workers, **kwargs
    )
    intermediates = pd.concat(
        {race_id: inters for race_id, (_, inters) in results.items()}, 
        axis=1
    )
    races_live_data = pd.concat(
        {race_id: live_data for race_id, (live_data, _) in results.items()}, 
        axis=0
    ).reset_index(drop=True)
    return races_live_data, intermediates


def calc_behind(live_time_data, gmt_speed=None, PGMT=1):
    if gmt_speed is None:
        distance = live_time_data.distanceTravelled.iloc[-1].max()
        gmt_speed = np.nanmax(live_time_data.avg_speed.values[
            live_time_data.distanceTravelled.values == distance 
        ])

    pace_speed = gmt_speed * PGMT
    t = live_time_data.index.values 
    pace_boat_d = pace_speed * t
    for c, d in live_time_data.distanceTravelled.items():
        live_time_data[("distanceFromPace", c)] = pace_boat_d - d
        live_time_data[("timeFromPace", c)] = t - np.interp(d, pace_boat_d, t)
        live_time_data[("PGMT", c)] = d / pace_boat_d

    return live_time_data


def get_current_data(live_data):
    current_data = live_data.iloc[[-1]].copy()
    current_data.PGMT = current_data.PGMT.applymap("{:.1%}".format)    
    current_data['time elapsed'] = current_data.time.max(1).map(utils.format_totalseconds)
    current_data.time = current_data.time.applymap(utils.format_totalseconds)
    return current_data.set_index('time elapsed').astype('string').T.unstack(1)


def get_race_livetracker(race_id, gmt=None, cached=True, live=False):
    endpoint = ('livetracker', "live") if live else ("livetracker",)
    data = get_worldrowing_data(*endpoint, race_id, cached=cached)
    
    if data and data['config']['lanes']:
        live_boat_data = parse_livetracker_data(data)
        intermediates = parse_intermediates_data(data)

        race_distance = data['config']['plot']['totalLength']

        gmt = gmt or find_world_best_time(
            race_id=race_id
        ).ResultTime.total_seconds()
        live_data = estimate_live_times(
            live_boat_data.reset_index(drop=True), 
            gmt=gmt,
            race_distance=race_distance
        )
        if "resultTime" in intermediates.columns:
            live_data = match_intermediate_times(
                live_data, get_intermediate_times(intermediates), race_distance
            )
    else:
        live_boat_data = pd.DataFrame([])
        intermediates = pd.DataFrame([])
        live_data = pd.DataFrame([])
        race_distance = None


    return live_data, live_boat_data, intermediates, race_distance


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
    for df in [results, intermediates]:
        if 'ResultTime' in df.columns:
            df.ResultTime = read_times(df.ResultTime)

    return results, intermediates


def parse_livetracker_raw_data(data, field, index=None):
    lanes = {
        lane['Lane']: lane 
        for lane in data['config']['lanes']
        if lane.get(field)
    }
    if not lanes:
        return pd.DataFrame([])
    
    lane_boat = pd.Series({
        lane_data['DisplayName']: lane
        for lane, lane_data in lanes.items()
    }).sort_values()

    if index is None:
        return pd.Series(
            {lane['DisplayName']: lane[field] for lane in lanes.values()},
            name=field
        )

    parsed = pd.concat(
        {
            lane['DisplayName']: pd.json_normalize(lane[field]).set_index(index) 
            for lane in lanes.values()
        }, 
        axis=1, 
        names=['boat', field]
    ).swaplevel(0, 1, 1)
    
    return parsed[
        pd.MultiIndex.from_product([
            parsed.columns.levels[0], 
            lane_boat.index
        ])
    ].copy()



def parse_livetracker_data(data):
    live_data = parse_livetracker_raw_data(data, 'live', 'id')
    if live_data.empty:
        return live_data 

    return live_data.rename(
        {c: c.split(".")[-1] for c in live_data.columns.levels[0]},
        axis=1, 
        level=0,
    )

def parse_intermediates_data(data):
    intermediates = parse_livetracker_raw_data(
        data, 'intermediates', 'distance.DisplayName'
    ).rename(columns={"raceConfig.value": "distance"})
    if 'ResultTime' in intermediates:
        for c, times in intermediates.ResultTime.items():
            intermediates[("ResultTime", c)] = pd.to_timedelta(times)

    return intermediates


def get_intermediate_times(intermediates):
    intermediate_results = intermediates[
        ["distance", "ResultTime"]
    ].stack(1).reset_index().set_index(
        ["distance", "boat"]
    ).ResultTime.unstack().sort_index(axis=1)
    distance_strs = intermediate_results.index
    distances = pd.Series(
        distance_strs.str.extract(
            r"([0-9]+)"
        )[0].astype(int).values,
        index=distance_strs,
        name='distance'
    ).sort_values()
    intermediate_results = intermediate_results.loc[distances.index]
    intermediate_results.index = distances
    intermediate_results.columns.name = 'country'
    return intermediate_results


def _parse_livetracker_data(data):
    total_length = data['config']['plot']
    lane_boat = {
        lane['Lane']: lane for lane in data['config']['lanes']
    }
    lane_cnt = {r: lane['DisplayName'] for r, lane in lane_boat.items()}
    countries = [lane_cnt[i] for i in sorted(lane_cnt)]

    id_cnt = {
        tracker['raceBoatId']: lane['DisplayName']
        for tracker, lane in zip(
            data['live'][0]['raceBoatTrackers'],
            data['config']['lanes']
        )
    }

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
            # cnt = lane_cnt[tracker['startPosition']]
            cnt = id_cnt[tracker['raceBoatId']]
            for key, boat_data in live_boat_data.items():
                boat_data[cnt].append(tracker[key])

    maxlen = max(
        max(map(len, live_data.values()))
        for live_data in live_boat_data.values()
    )
    if not live_data['distanceOfLeaderFromFinish']:
        total_distance = live_data['distanceOfLeader']
        for cnt, dists in live_boat_data['distanceTravelled'].items():
            for key, live_data in live_boat_data.items():
                cnt_data = live_data[cnt]
                cnt_len = len(cnt_data)
                if key == 'distanceTravelled':
                    cnt_data.extend([total_distance] * (maxlen - cnt_len + 1))
                else:
                    cnt_data.extend(cnt_data[-1:] * (maxlen - cnt_len + 1))
    else:
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


def estimate_times(live_boat_data, finish_distance=2000):
    live_data = live_boat_data.reset_index(drop=True).copy()
    
    for c, i in (live_data.distanceTravelled == finish_distance).idxmax(0).items():
        live_data.loc[i + 1:, (slice(None), c)] = np.nan 

    distance = live_data.distanceTravelled
    speed = live_data.metrePerSecond.copy()
    speed.loc[0] = speed.loc[1]

    Dx = distance.diff()
    V = (speed + speed.shift())/2
    dT = (Dx/V).mean(1).fillna(0)
    T = dT.cumsum()

    keep = dT > 0
    race_data = live_data.loc[keep]
    race_data.index = T[keep] 
    race_data.index.name = 'time'
    race_data = race_data.fillna(method='ffill').copy()
    for c, dtype in race_data.dtypes.groupby(level=0).first().items():
        if dtype is np.dtype('float') and c not in {"metrePerSecond"}:
            race_data[c] = race_data[c].fillna(0).astype(int)

    return race_data

def estimate_live_times(live_boat_data, gmt, race_distance=2000):
    countries = live_boat_data.columns.levels[1]
    distance_travelled = live_boat_data.distanceTravelled.fillna(method='ffill')
    speed = live_boat_data.metrePerSecond.fillna(method='ffill')

    live_data = pd.concat({
        "distanceTravelled": live_boat_data.distanceTravelled.fillna(race_distance).astype(int),
        "metrePerSecond": live_boat_data.metrePerSecond.fillna(method='ffill'),
        "strokeRate": live_boat_data.strokeRate.fillna(method='ffill'),
        "currentPosition": live_boat_data.currentPosition.fillna(method='ffill').astype(int),
        'distanceFromLeader': live_boat_data.distanceFromLeader.fillna(method='ffill').astype(int),
    }, axis=1)
    
    n_countries = len(countries)
    distances = np.c_[
        np.zeros(n_countries), distance_travelled.values.T
    ].T
    boat_diffs = np.diff(distances, axis=0)
    boat_times = boat_diffs / speed

    
    mean_time_diffs = np.ma.masked_array(
        boat_times, mask=boat_diffs == 0
    ).mean(1).data
    for cnt in countries:
        live_data['time', cnt] = np.where(
            live_data.distanceTravelled[cnt] == race_distance,
            boat_times[cnt], 
            mean_time_diffs
        ).cumsum()

    for cnt in countries:
        live_data['PGMT', cnt] = (
            live_data.distanceTravelled[cnt] / race_distance
            * gmt / live_data.time[cnt]
        )

    update_time_from_leader(live_data)

    return live_data

def update_time_from_leader(live_data):    
    distance_travelled = live_data.distanceTravelled
    countries = distance_travelled.columns
    times = live_data.time[countries]

    leader = distance_travelled.combine(
        - times, lambda d, t: d.combine(t, lambda *args: args)
    )[countries].values.argmax(1)
    leader_distance = distance_travelled.values[
        np.arange(distance_travelled.shape[0]), leader
    ]
    leader_time = times.values[np.arange(distance_travelled.shape[0]), leader]
    dmax = leader_distance.max()
    imax = leader_distance.searchsorted(dmax) + 1

    for cnt in countries:
        dist = distance_travelled[cnt]
        delta = (
            times[cnt].values
            - np.interp(
                dist, leader_distance[:imax], leader_time[:imax]
            )
        )
        jlast = min(
            dist.values.searchsorted(dmax),
            len(delta) - 1
        )
        delta[jlast:] = delta[jlast]
        live_data[('timeFromLeader', cnt)] = np.clip(delta, 0, None)

    return live_data

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


def estimate_live_intermediates(live_data, distances):
    return pd.DataFrame({
        cnt: pd.Series(
            np.interp(
                distances, 
                live_data.distanceTravelled[cnt], 
                live_data.time[cnt]
            ), 
            distances, 
        )
        for cnt in live_data.columns.levels[1]
    })

def match_intermediate_times(live_data, intermediates, race_distance):
    inters = intermediates.apply(lambda t: t.dt.total_seconds())
    adj_live = live_data.copy()
    diff = estimate_live_intermediates(live_data, inters.index) - inters

    mean_shift = diff.mean(axis=1)
    start_shift = mean_shift[mean_shift.index != race_distance].mean()
    adj_live.time -= start_shift
    mean_shift -= start_shift

    for cnt in inters.columns:
        adj_live.loc[:, ('time', cnt)] -= np.interp(
            adj_live.distanceTravelled[cnt], 
            mean_shift.index, 
            mean_shift,
            left=0, 
            right=0, 
        )

    diff = estimate_live_intermediates(adj_live, inters.index) - inters

    for cnt, final_diff in diff.reindex([race_distance]).iloc[0].items():
        if np.isfinite(final_diff):
            i = adj_live.distanceTravelled[cnt].searchsorted(race_distance)
            adj_live.loc[i:, ("time", cnt)] -= final_diff

    diff = estimate_live_intermediates(adj_live, inters.index) - inters

    for cnt in diff.columns:
        x, y = inters.index.values, inters[cnt].values 
        i = adj_live.distanceTravelled[cnt].searchsorted(x, side='left')
        if i[-1] not in adj_live.index:
            i, x, y = i[:-1], x[:-1], y[:-1]
            
        x0, y0 = adj_live.distanceTravelled[cnt][i-1].values, adj_live.time[cnt][i - 1].values
        x1, y1 = adj_live.distanceTravelled[cnt][i].values, adj_live.time[cnt][i].values
        shift = (x - x0) - (y - y0) / (y1 - y0) * (x1 - x0)

        adj_live.loc[:, ('distanceTravelled', cnt)] += np.interp(
            adj_live.distanceTravelled[cnt], 
            np.r_[0, np.c_[x0, x1].ravel()], 
            np.r_[0, np.c_[shift, shift].ravel()],
            left=0, 
            right=0, 
        )
    
    update_time_from_leader(adj_live)
    return adj_live

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

def load_races_data(save_path, finished_races=(), download=True):
    save_path = Path(save_path)
    downloaded = {
        (p.parent.name, p.stem.replace("-", "/")): (p,) 
        for p in save_path.glob("*/*.json")
    }
    logger.info(
        '%d races livetracker data already downloaded, loading from "%s"', 
        len(downloaded), save_path 
    )
    race_live_data, errors = utils.map_concurrent(
        lambda p: json.load(open(p, 'r')), downloaded
    )
    if len(finished_races) and download:
        to_download = {
            k: ("livetracker", finished_races.loc[k, 'id'])
            for k in finished_races.index[:].difference(race_live_data.keys())
        }
        if to_download:
            logger.info(
                "downloading %d races from World rowing", 
                len(to_download) 
            )

            downloaded, errors = utils.map_concurrent(
                api.get_worldrowing_data,
                to_download, 
                max_workers=4,
                requests_kws=(("timeout", 5.),)
            )
            
            logger.info(
                "saving %d races from World Rowing to %s", 
                len(downloaded), save_path 
            )
            for (event_name, race_name), live_data in tqdm(downloaded.items()):
                event_path = save_path / event_name
                event_path.mkdir(parents=True, exist_ok=True)
                name = race_name.replace("/", "-")
                file_path = event_path / f"{name}.json"
                with open(file_path, 'w') as f:
                    json.dump(live_data, f)

            race_live_data.update(downloaded)
        else:
            logger.info("all races already downloaded")

    return race_live_data 

def load_competition_data(
        competition=None, 
        save_path="live_tracker", 
        competition_path=None, 
        download=True
):
    save_path = Path(save_path)
    if competition_path is None:
        competition = (
            api.get_most_recent_competition() if competition is None else competition
        )
        competition_path = save_path / competition.DisplayName
        competition_path.mkdir(exist_ok=True, parents=True)
        competition.to_frame().to_json(competition_path / "competition.json")
    else:
        competition_path = Path(competition_path)
        competition = pd.read_json(
            competition_path / "competition.json"
        ).iloc[:, 0]
    
    if download:
        logger.info(
            "downloading up-to-date race for %s from World Rowing",
            competition.DisplayName
        )
        races = api.get_competition_races(competition.name, cached=False)
        events = api.get_competition_events(competition.name)
        wbts = api.get_world_best_times()

        events['BoatClass'] = api.get_boat_types().DisplayName[
            events.boatClassId
        ].values
        events['worldBestTime'] = wbts.reindex(events.BoatClass).ResultTime.values 
        races = pd.merge(
            races.reset_index(), 
            events, 
            how='left', 
            left_on='eventId', 
            right_on='id', 
            suffixes=("", "_event")
        ).rename(columns={
            "DisplayName_event": "EventName",
        })

        finished_races = races[
            races['raceStatus.DisplayName'] == 'Official'
        ].sort_values('Date')
        finished_races.to_json(
            competition_path / "finished_races.json"
        )
    else:
        races_meta_path = competition_path / "finished_races.json"
        logger.info(
            'loading saved data for %s from "%s"',
            competition.DisplayName, 
            races_meta_path, 
        )
        finished_races = pd.read_json(races_meta_path)
        finished_races.DateString = pd.to_datetime(
            finished_races.DateString, unit='ms'
        )
        finished_races.worldBestTime = pd.to_timedelta(
            finished_races.worldBestTime, unit='ms'
        )

    logger.info(
        "loading %d races for competition %s", 
        len(finished_races), competition.DisplayName
    )
    race_live_data = load_races_data(
        save_path / competition.DisplayName, 
        finished_races.set_index(["EventName", "DisplayName"]),
        download=download
    )

    return competition, finished_races, race_live_data