
import datetime
from functools import partial
import json
from typing import NamedTuple, Callable, Optional, Dict

import numpy as np
import pandas as pd
# from scipy import sparse
from scipy import stats

from sklearn import metrics

import jax
from jax import numpy as jnp, tree_map
from jax.experimental import sparse
from jax.scipy.linalg import solve_triangular
from jax.scipy import linalg
import jaxopt

import haiku as hk

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.autonotebook import tqdm

from rowing.world_rowing import api, utils, fields
from rowing.model.gp import kernels, utils as gp_utils


def get_athlete_kernel():
    return kernels.SumKernel(
        # kernels.IntSEKernel(name="athlete_intse_0"),
        # kernels.IntSEKernel(name="athlete_int_kernel2"),
        kernels.SEKernel(name="athlete_se_0"),
        # kernels.SEKernel(name="athlete_se_1"),
        # kernels.Matern12(name="athlete_matern12"),
        # kernels.Matern32(name="athlete_matern32"),
        kernels.Matern52(name="athlete_matern52"),
        kernels.Bias(name='athlete_bias')
    )


def get_race_kernel():
    return kernels.SumKernel(
        kernels.Matern12(name='race_matern12_0'),
        kernels.SEKernel(name='race_kernel0'),
        # kernels.SEKernel(name='race_kernel1'),
        # kernels.Bias(name='race_bias'),
    )


def get_weather_kernel(n_features):
    return kernels.SumKernel(
        kernels.SEKernel(name='weather_se_0', shape=(n_features,)),
        kernels.DotProduct(name='weather_dot_0'),
        kernels.DotProduct(name='weather_dot_1')**2,
        # kernels.DotProduct(name='weather_dot_2')**3,
        # kernels.Matern32(name='weather_matern32_0', shape=(n_features,)),
        # kernels.Matern52(name='weather_matern52_0', shape=(n_features,)),
    )


GetKernel = Callable[[], kernels.AbstractKernel]
GetKernelD = Callable[[int], kernels.AbstractKernel]


class AthleteModel(NamedTuple):
    years: jax.Array
    year0: jax.Array
    W_athlete: jax.Array
    gram_athlete: jax.Array
    athlete_kernel: GetKernel = get_athlete_kernel
    metadata: Optional[Dict] = None

    def get_kernels(self):
        years = self.years - self.year0
        K_athlete = self.athlete_kernel().K(years, years) * self.gram_athlete
        return K_athlete,


def boatclass_kernel(K):
    boatclass_var = jnp.exp(hk.get_parameter(
        fields.BoatType, [], init=jnp.zeros))
    return boatclass_var * K


class RaceModel(NamedTuple):
    hours: jax.Array
    W_venue: jax.Array
    W_boatclass: jax.Array
    W_lane: jax.Array
    gram_venue: jax.Array
    gram_boatclass: jax.Array
    gram_lane: jax.Array
    race_kernel: GetKernel = get_race_kernel
    lane_kernel: Optional[GetKernel] = None
    metadata: Optional[Dict] = None

    def get_kernels(self):
        times = self.hours
        K_race_times = self.race_kernel().K(times, times) * self.gram_venue
        if self.lane_kernel:
            gram_lane = jnp.where(
                jnp.isfinite(self.gram_lane), self.gram_lane, 0
            )
            K_lane = jnp.where(
                jnp.isfinite(self.gram_lane),
                self.lane_kernel().K(times, times)
                * self.gram_venue
                * gram_lane,
                0
            )
            K_race_times += K_lane

        K_boatclass = boatclass_kernel(self.gram_boatclass)

        return K_race_times, K_boatclass


class RaceWeatherModel(NamedTuple):
    hours: jax.Array
    weather: jax.Array
    W_venue: jax.Array
    W_boatclass: jax.Array
    gram_venue: jax.Array
    gram_boatclass: jax.Array

    y: Optional[jax.Array] = None
    race_kernel: GetKernel = get_race_kernel
    weather_kernel: GetKernelD = get_weather_kernel

    get_full_kernel = gp_utils.get_full_kernel
    get_jitter_kernel = gp_utils.get_jitter_kernel
    gp_system = gp_utils.gp_system
    loss = gp_utils.loss

    def get_kernels(self):
        K_boatclass = boatclass_kernel(self.gram_boatclass)

        kernels = K_boatclass,

        times = self.hours
        if self.race_kernel:
            K_race_times = self.race_kernel().K(times, times) * self.gram_venue
            kernels += K_race_times,

        if self.weather_kernel:
            weather = self.weather
            K_weather = self.weather_kernel(
                weather.shape[1]).K(weather, weather)
            kernels += K_weather,

        return kernels

    @classmethod
    def from_conditions(cls, race_conditions, weather_cols, **kwargs):
        start_times = race_conditions.Race_Start_utc
        order = race_conditions.race_id

        weather = jnp.array(race_conditions[weather_cols].values)

        first_time = start_times.min()
        times = (start_times - first_time).dt.total_seconds().values
        hours = times / 60 / 60

        weights = {
            f: race_conditions.groupby([
                "race_id", f
            ]).size().unstack(level=1, fill_value=0).loc[order]
            for f in [
                fields.Day,
                "race_event_competition_venueId",
                fields.race_boatClass,
                fields.BoatType,
            ]
        }
        Ws = {k: jnp.array(df.values) for k, df in weights.items()}
        grams = {k: W @ W.T for k, W in Ws.items()}

        return cls(
            hours=hours,
            weather=weather,
            gram_venue=grams['race_event_competition_venueId'],
            gram_boatclass=grams['Boat Class'],
            y=jnp.array(race_conditions.PGMT.values),
            **kwargs
        )


class PerformanceModel(NamedTuple):
    athlete_model: AthleteModel
    race_model: RaceModel
    y: jax.Array
    metadata: Optional[Dict] = None

    get_full_kernel = gp_utils.get_full_kernel
    get_jitter_kernel = gp_utils.get_jitter_kernel
    gp_system = gp_utils.gp_system
    loss = gp_utils.loss

    def get_kernels(self):
        return (
            self.athlete_model.get_kernels()
            + self.race_model.get_kernels()
        )

    @classmethod
    def from_data(cls, results, seats, athletes, **kwargs):
        seats = seats.join(
            1 / seats.groupby(level=0).size().rename("seat_weight"),
            on='athletes_raceBoatId'
        ).join(
            results['Boat Type'], on='athletes_raceBoatId'
        )
        boat_order = results.index
        athlete_order = athletes.index

        weights = {
            f: results.groupby([
                "raceBoats_id", f
            ]).size().unstack(level=1, fill_value=0).loc[boat_order]
            for f in [
                fields.Day,
                "race_event_competition_venueId",
                fields.race_boatClass,
                fields.BoatType,
            ]
        }
        weights['athlete'] = seats.seat_weight.unstack(
            level=1, fill_value=0).loc[boat_order, athlete_order]
        weights['lane'] = (
            results.Lane -
            results.groupby("race_id").Lane.mean().loc[results.race_id].values
        ).loc[boat_order].to_frame()

        Ws = {k: jnp.array(df.values) for k, df in weights.items()}
        grams = {k: W @ W.T for k, W in Ws.items()}

        start_times = results[fields.race_Date]
        first_time = start_times.min()
        last_time = start_times.max()

        first_year = first_time.year + first_time.day_of_year / 365.25
        last_year = last_time.year + last_time.day_of_year / 365.25

        times = (start_times - first_time).dt.total_seconds().values
        hours = times / 60 / 60
        years = first_year + (last_year - first_year) * \
            (times - times.min())/(times.max() - times.min())

        year0 = first_year - 2

        race_model = RaceModel(
            hours=hours,
            W_boatclass=Ws["Boat Class"],
            W_venue=Ws['race_event_competition_venueId'],
            W_lane=Ws['lane'],
            gram_venue=grams['race_event_competition_venueId'],
            gram_boatclass=grams['Boat Class'],
            gram_lane=grams['lane'],
        )
        athlete_model = AthleteModel(
            years=years,
            year0=year0,
            W_athlete=Ws["athlete"],
            gram_athlete=grams['athlete'],
        )

        return cls(
            athlete_model=athlete_model,
            race_model=race_model,
            y=results.PGMT.values,
            metadata={
                "weights": weights,
            },
        )


class CompetitionModel(NamedTuple):
    hours: np.ndarray
    years: np.ndarray
    year0: np.ndarray
    W_venue: np.ndarray
    W_athlete: np.ndarray
    W_boatclass: np.ndarray
    W_lane: np.ndarray
    y: np.ndarray
    gram_venue: np.ndarray
    gram_athlete: np.ndarray
    gram_boatclass: np.ndarray
    gram_lane: np.ndarray
    athlete_kernel: GetKernel = get_athlete_kernel
    race_kernel: GetKernel = get_race_kernel
    lane_kernel: Optional[GetKernel] = get_race_kernel
    metadata: Optional[Dict] = None

    get_full_kernel = gp_utils.get_full_kernel
    get_jitter_kernel = gp_utils.get_jitter_kernel
    gp_system = gp_utils.gp_system
    loss = gp_utils.loss

    @classmethod
    def from_data(cls, results, seats, athletes, **kwargs):
        seats = seats.join(
            1 / seats.groupby(level=0).size().rename("seat_weight"),
            on='athletes_raceBoatId'
        ).join(
            results['Boat Type'], on='athletes_raceBoatId'
        )
        boat_order = results.index
        athlete_order = athletes.index

        weights = {
            f: results.groupby([
                "raceBoats_id", f
            ]).size().unstack(level=1, fill_value=0).loc[boat_order]
            for f in [
                fields.Day,
                "race_event_competition_venueId",
                fields.race_boatClass,
                fields.BoatType,
            ]
        }
        weights['athlete'] = seats.seat_weight.unstack(
            level=1, fill_value=0).loc[boat_order, athlete_order]
        # Make lanes 0 mean per race.
        weights['lane'] = (
            results.Lane -
            results.groupby("race_id").Lane.mean().loc[results.race_id].values
        ).loc[boat_order].to_frame()

        Ws = {k: jnp.array(df.values) for k, df in weights.items()}
        grams = {k: W @ W.T for k, W in Ws.items()}

        start_times = results[fields.race_Date]
        first_time = start_times.min()
        last_time = start_times.max()

        first_year = first_time.year + first_time.day_of_year / 365.25
        last_year = last_time.year + last_time.day_of_year / 365.25

        times = (start_times - first_time).dt.total_seconds().values
        hours = times / 60 / 60
        years = first_year + (last_year - first_year) * \
            (times - times.min())/(times.max() - times.min())

        year0 = first_year - 2
        return cls(
            hours=hours,
            years=years,
            year0=year0,
            W_venue=Ws['race_event_competition_venueId'],
            W_athlete=Ws["athlete"],
            W_boatclass=Ws["Boat Class"],
            W_lane=Ws['lane'],
            y=results.PGMT.values,
            gram_venue=grams['race_event_competition_venueId'],
            gram_athlete=grams['athlete'],
            gram_boatclass=grams['Boat Class'],
            gram_lane=grams['lane'],
            metadata={
                "weights": weights,
            },
            **kwargs
        )

    def get_kernels(self):
        boatclass_var = jnp.exp(hk.get_parameter(
            fields.BoatType, [], init=jnp.zeros))

        times = self.hours
        years = self.years - self.year0

        K_race_times = self.race_kernel().K(times, times) * self.gram_venue
        if self.lane_kernel:
            K_race_times += self.lane_kernel().K(times, times) * \
                self.gram_venue * self.gram_lane
        K_athlete_times = self.athlete_kernel().K(years, years) * self.gram_athlete
        K_boatclass = boatclass_var * self.gram_boatclass

        return (
            K_race_times,
            K_athlete_times,
            K_boatclass,
        )

    get_full_kernel = gp_utils.get_full_kernel
    get_jitter_kernel = gp_utils.get_jitter_kernel
    gp_system = gp_utils.gp_system
    loss = gp_utils.loss

    # def get_full_kernel(self):
    #     K_race_times, K_athlete_times, K_boatclass = self.get_kernels()
    #     return (
    #         K_race_times
    #         + K_athlete_times
    #         + K_boatclass
    #     )

    # def get_jitter_kernel(self):
    #     K = self.get_full_kernel()
    #     race_var = jnp.exp(hk.get_parameter(
    #         "race_logvar", [], init=jnp.zeros, dtype=jnp.float64))
    #     K_noise = np.eye(len(K)) * race_var
    #     return K + K_noise

    # def gp_system(self):
    #     K = self.get_jitter_kernel()
    #     y = self.y
    #     return gp_utils.GPSystem.from_gram(K, y)

    # def loss(self):
    #     return self.gp_system().loss()


def filter_results(
    senior_data,
    years=None, keep_phases=None, min_races=None,
    min_racesize=None, min_pgmt=None, max_pgmt=1,
    keep_athletes=None, **kwargs,
):
    results = senior_data['results']
    athletes = senior_data['athletes']
    seats = senior_data['seats']

    sel_results = results
    filtered = pd.Series(False, athletes.athletes_personId)
    filtered_raceBoats = set()
    for i in range(20):
        sel = (
            ~ results.raceBoats_id.isin(filtered_raceBoats)
            & np.isfinite(results.PGMT)
            & results['Race Start'].notna()
            & results.raceBoats_id.isin(seats.athletes_raceBoatId)
        )
        if years:
            sel &= results.year.isin(years)
        if keep_phases:
            sel &= results.Phase.isin(keep_phases)
        if min_pgmt:
            sel &= results.PGMT > min_pgmt
        if max_pgmt:
            sel &= results.PGMT <= max_pgmt
        if min_racesize:
            sel &= (
                sel_results.groupby("race_id").size() > min_racesize
            ).reindex(results.race_id, fill_value=False).values
        for k, filter in kwargs.items():
            sel &= filter(results[k])

        sel_results = results[sel].set_index(
            "raceBoats_id").sort_values('Race Start')
        sel_seats = seats[
            seats.athletes_raceBoatId.isin(sel_results.index)
            & (seats.athletes_boatPosition != "c")
        ].set_index([
            'athletes_raceBoatId', 'athletes_personId',
        ]).sort_index()

        update = sel_seats.groupby(level=1).size() < min_races
        filtered.update(update)
        if keep_athletes is not None:
            filtered[keep_athletes] = False
        if not update.any():
            break

        filtered_raceBoats = seats.athletes_raceBoatId[
            seats.athletes_personId.isin(filtered.index[filtered])
        ]

    sel_athletes = athletes[
        athletes.athletes_personId.isin(sel_seats.index.levels[1])
    ].set_index("athletes_personId").sort_index()

    return {
        "athletes": sel_athletes,
        "results": sel_results,
        "seats": sel_seats,
        "competitions": senior_data['competitions']
    }


def get_full_kernel(self):
    kernels = self.get_kernels()
    return sum(kernels)


def get_jitter_kernel(self):
    K = self.get_full_kernel()
    race_var = jnp.exp(hk.get_parameter(
        "log_noise", [], init=jnp.zeros, dtype=jnp.float64))
    K_noise = jnp.eye(len(K)) * race_var
    return K + K_noise


def gp_system(self):
    K = self.get_jitter_kernel()
    y = self.y
    return gp_utils.GPSystem.from_gram(K, y)


def loss(self):
    return self.gp_system().loss()


# class CompetitionModel(NamedTuple):
#     hours0: np.ndarray
#     hours1: np.ndarray
#     years0: np.ndarray
#     years1: np.ndarray
#     y: np.ndarray
#     gram_venue: np.ndarray
#     gram_athlete: np.ndarray
#     gram_boatclass: np.ndarray
#     athlete_kernel: Callable[[], kernels.AbstractKernel] = get_athlete_kernel
#     race_kernel: Callable[[], kernels.AbstractKernel] = get_race_kernel
#     metadata: Optional[Dict] = None

#     get_full_kernel = get_full_kernel
#     get_jitter_kernel = get_jitter_kernel
#     gp_system = gp_system
#     loss = loss

#     @classmethod
#     def from_data(cls, results, seats, athletes, **kwargs):
#         seats = seats.join(
#             1 / seats.groupby(level=0).size().rename("seat_weight"),
#             on='athletes_raceBoatId'
#         ).join(
#             results['Boat Type'], on='athletes_raceBoatId'
#         )
#         boat_order = results.index
#         athlete_order = athletes.index

#         weights = {
#             f: results.groupby([
#                 "raceBoats_id", f
#             ]).size().unstack(level=1, fill_value=0).loc[boat_order]
#             for f in [
#                 fields.Day,
#                 "race_event_competition_venueId",
#                 fields.race_boatClass,
#                 fields.BoatType,
#             ]
#         }
#         weights['athlete'] = seats.seat_weight.unstack(
#             level=1, fill_value=0).loc[boat_order, athlete_order]

#         Ws = {k: jnp.array(df.values) for k, df in weights.items()}
#         grams = {k: W @ W.T for k, W in Ws.items()}

#         start_times = results[fields.race_Date]
#         first_time = start_times.min()
#         last_time = start_times.max()
#         first_year = first_time.year + first_time.day_of_year / 365.25
#         last_year = last_time.year + last_time.day_of_year / 365.25
#         times = (start_times - first_time).dt.total_seconds().values
#         hours = times / 60 / 60
#         years = (last_year - first_year) * \
#             (times - times.min())/(times.max() - times.min())

#         return cls(
#             hours0=hours,
#             hours1=hours,
#             years0=years,
#             years1=years,
#             y=results.PGMT.values,
#             gram_venue=grams['race_event_competition_venueId'],
#             gram_athlete=grams['athlete'],
#             gram_boatclass=grams['Boat Class'],
#             metadata={
#                 "weights": weights,
#             },
#             **kwargs
#         )

#     def get_kernels(self):
#         boatclass_var = jnp.exp(hk.get_parameter(
#             fields.BoatType, [], init=jnp.zeros))

#         K_race_times = self.race_kernel().K(self.hours0, self.hours1) * self.gram_venue
#         K_athlete_times = self.athlete_kernel().K(
#             self.years0, self.years1) * self.gram_athlete
#         K_boatclass = boatclass_var * self.gram_boatclass
#         return (
#             K_race_times, K_athlete_times, K_boatclass,
#         )
