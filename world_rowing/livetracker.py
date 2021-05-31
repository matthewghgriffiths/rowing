
from functools import cached_property

import numpy as np
import pandas as pd
from scipy import stats

from .api import (
    get_worldrowing_data, get_race_results, get_worldrowing_record,
    find_world_best_time, INTERMEDIATE_FIELDS
)
from .utils import extract_fields, format_yaxis_splits

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
    ):
        self.race_id = race_id
        
        self.gmt = gmt or find_world_best_time(
            race_id=race_id
        ).ResultTime.total_seconds()   
        
        self._colors = colors
        
        self.live_data = live_data 
        self.results = results
        self.intermediates = intermediates
        
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
        if self.intermediates is not None:
            intermediate_results = pd.merge(
                self.intermediates[
                    ['raceBoatId', 'distance', 'ResultTime']
                ], 
                self.results[
                    ['id', 'DisplayName']],
                left_on='raceBoatId', 
                right_on='id', 
                how='left'
            ).set_index(
                ['DisplayName', 'distance']
            ).ResultTime.dt.total_seconds().unstack()
            distance_strs = intermediate_results.columns
            distances = pd.Series(
                distance_strs.str.extract(
                    r"([0-9]+)"
                )[0].astype(int).values,
                index=distance_strs,
                name='distance'
            ).sort_values()
            intermediate_results = intermediate_results[distances.index]
            intermediate_results.columns = distances
            intermediate_results.index.name = 'country'
            return intermediate_results
        
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
        self.live_data, self.results, self.intermediates = \
            get_race_livetracker(
                self.race_id, 
                gmt=self.gmt,
                cached=False,
        )

    def bar(self, y, bottom=None, yerr=None, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        bar = ax.bar(
            self.lane_country.index,
            y[self.lane_country],
            bottom=bottom, 
            yerr=yerr, 
            tick_label=self.lane_country,
            color=self.country_colors[self.lane_country],
            **kwargs
        )
        return bar

    def violin(
            self, y_dens, ax=None, width=0.8, alpha=0.6, outline=True, 
            set_xticks=True, outline_kws=None, **kwargs
    ):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        violins = {}
        lines = {}
        if width:
            y_dens = y_dens / y_dens.max(0) * width /2

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
                lines[cnt] = ax.plot(
                    np.r_[-y_dens[cnt] + x0, y_dens[cnt].iloc[::-1] + x0],
                    np.r_[y_dens.index, y_dens.index[::-1]],
                    color=self.country_colors[cnt],
                    label=cnt,
                    **(outline_kws or {})
                )
        
        if set_xticks:
            ax.set_xticks(self.lane_country.index)
            ax.set_xticklabels(self.lane_country)
        return ax, violins, lines

    def plot_finish_times(
            self, pred_finish, finish_std, n_std=3, 
            ax=None, **kwargs
    ):
        ylim = (pred_finish.min() - 4 * finish_std, 
        pred_finish.max() + 4 * finish_std)
        y = np.linspace(*ylim, 1000)
        y_dens = pd.DataFrame({
            cnt: stats.norm(
                loc=pred_finish[cnt], 
                scale=finish_std
            ).pdf(y)
            for cnt in self.lane_country
        }, 
            index=y
        )
        ax, violins, lines = self.violin(y_dens, ax=ax, **kwargs)
        ax.set_ylim(*ylim)
        ax.invert_yaxis()
        format_yaxis_splits(ax)
        return ax, violins, lines
        
    def plot(self, distance, y, ax=None, maxdistance=None, **kwargs):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        lines = {}
        if not isinstance(distance, (pd.DataFrame, dict)):
            distance = {cnt: distance for cnt in y.columns}

        for cnt in y.columns:
            lines[cnt] =  ax.plot(
                distance[cnt], 
                y[cnt], 
                label=cnt, 
                color=self.country_colors[cnt], 
                **kwargs
            )
        ax.set_xlim(0, maxdistance or 2000)
        ax.set_xlabel('distance / m')
        
        return ax, lines

    def plot_uncertainty(
            self, distance, y, yerr, ax=None, alpha=0.2,
            maxdistance=None, fill_kws=None, **kwargs
    ):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        lines = {}
        collections = {}

        if not isinstance(distance, (pd.DataFrame, dict)):
            distance = {cnt: distance for cnt in y.columns}
        if not isinstance(yerr, (pd.DataFrame, dict)):
            yerr = {cnt: yerr for cnt in y.columns}

        for cnt in y.columns:
            lines[cnt] =  ax.plot(
                distance[cnt], 
                y[cnt], 
                label=cnt, 
                color=self.country_colors[cnt], 
                **kwargs
            )
            collections[cnt] =  ax.fill_between(
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
        
        return ax, lines, collections

    
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
    for df in [results, intermediates]:
        if 'ResultTime' in df.columns:
            minfmt = df.ResultTime.str.match(
                r"[0-9]+:[0-9][0-9]\.[0-9]")
            df.loc[minfmt, 'ResultTime'] = "0:" + df.ResultTime[minfmt]
            df.loc[:, 'ResultTime'] = pd.to_timedelta(df.ResultTime)

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