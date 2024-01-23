
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
        kernels.IntSEKernel(name="athlete_kernel"),
        kernels.IntSEKernel(name="athlete_int_kernel2"),
        kernels.SEKernel(name="athlete_se_kernel"),
        kernels.SEKernel(name="athlete_se_kernel2"),
        kernels.Bias(name='athlete_bias')
    )


def get_race_kernel():
    return kernels.SumKernel(
        kernels.SEKernel(name='race_kernel'),
        kernels.SEKernel(name='race_kernel2'),
        kernels.Bias(name='race_bias'),
    )


class CompetitionModel(NamedTuple):
    hours: np.ndarray
    years: np.ndarray
    year0: np.ndarray
    W_venue: np.ndarray
    W_athlete: np.ndarray
    W_boatclass: np.ndarray
    y: np.ndarray
    gram_venue: np.ndarray
    gram_athlete: np.ndarray
    gram_boatclass: np.ndarray
    athlete_kernel: Callable[[], kernels.AbstractKernel] = get_athlete_kernel
    race_kernel: Callable[[], kernels.AbstractKernel] = get_race_kernel
    metadata: Optional[Dict] = None

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
            y=results.PGMT.values,
            gram_venue=grams['race_event_competition_venueId'],
            gram_athlete=grams['athlete'],
            gram_boatclass=grams['Boat Class'],
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
        K_athlete_times = self.athlete_kernel().K(years, years) * self.gram_athlete
        K_boatclass = boatclass_var * self.gram_boatclass

        return (
            K_race_times,
            K_athlete_times,
            K_boatclass,
        )

    def get_full_kernel(self):
        K_race_times, K_athlete_times, K_boatclass = self.get_kernels()
        return (
            K_race_times
            + K_athlete_times
            + K_boatclass
        )

    def get_jitter_kernel(self):
        K = self.get_full_kernel()
        race_var = jnp.exp(hk.get_parameter(
            "race_logvar", [], init=jnp.zeros, dtype=jnp.float64))
        K_noise = np.eye(len(K)) * race_var
        return K + K_noise

    def gp_system(self):
        K = self.get_jitter_kernel()
        y = self.y
        return gp_utils.GPSystem.from_gram(K, y)

    def loss(self):
        return self.gp_system().loss()


class CompetitionModel2(NamedTuple):
    hours0: np.ndarray
    hours1: np.ndarray
    years0: np.ndarray
    years1: np.ndarray
    y: np.ndarray
    gram_venue: np.ndarray
    gram_athlete: np.ndarray
    gram_boatclass: np.ndarray
    athlete_kernel: Callable[[], kernels.AbstractKernel] = get_athlete_kernel
    race_kernel: Callable[[], kernels.AbstractKernel] = get_race_kernel
    metadata: Optional[Dict] = None

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

        Ws = {k: jnp.array(df.values) for k, df in weights.items()}
        grams = {k: W @ W.T for k, W in Ws.items()}

        start_times = results[fields.race_Date]
        first_time = start_times.min()
        last_time = start_times.max()
        first_year = first_time.year + first_time.day_of_year / 365.25
        last_year = last_time.year + last_time.day_of_year / 365.25
        times = (start_times - first_time).dt.total_seconds().values
        hours = times / 60 / 60
        years = (last_year - first_year) * \
            (times - times.min())/(times.max() - times.min())

        return cls(
            hours0=hours,
            hours1=hours,
            years0=years,
            years1=years,
            y=results.PGMT.values,
            gram_venue=grams['race_event_competition_venueId'],
            gram_athlete=grams['athlete'],
            gram_boatclass=grams['Boat Class'],
            metadata={
                "weights": weights,
            },
            **kwargs
        )

    def get_kernels(self):
        boatclass_var = jnp.exp(hk.get_parameter(
            fields.BoatType, [], init=jnp.zeros))

        K_race_times = self.race_kernel().K(self.hours0, self.hours1) * self.gram_venue
        K_athlete_times = self.athlete_kernel().K(
            self.years0, self.years1) * self.gram_athlete
        K_boatclass = boatclass_var * self.gram_boatclass
        return (
            K_race_times, K_athlete_times, K_boatclass,
        )

    def get_full_kernel(self):
        K_race_times, K_athlete_times, K_boatclass = self.get_kernels()
        return (
            K_race_times + K_athlete_times + K_boatclass
        )

    def get_jitter_kernel(self):
        K = self.get_full_kernel()
        race_var = jnp.exp(hk.get_parameter(
            "race_logvar", [], init=jnp.zeros, dtype=jnp.float64))
        K_noise = np.eye(len(K)) * race_var
        return K + K_noise

    def gp_system(self):
        K = self.get_jitter_kernel()
        y = self.y
        return gp_utils.GPSystem.from_gram(K, y)

    def loss(self):
        return self.gp_system().loss()

# def predict_feature(params, data, field, system=None):
#     a, Ly, L, y = gp_system(params, data) if system is None else system
#     var = np.exp(params['~'][field])
#     feature = data['features'][field]
#     return (a @ feature) * var


# def predict_race_pgmts(params, data, system=None):
#     a, Ly, L, y = gp_system(params, data) if system is None else system
#     K_race_times, K_athlete_times, K_boatclass = gp_utils.transform(
#         get_kernels
#     ).apply(params, data)
#     return (K_race_times + K_boatclass) @ a


# def predict_athlete_scores(params, data, pred_years, system=None):
#     a, Ly, L, y = gp_system(params, data) if system is None else system

#     features = data['features']
#     athlete_weights = features[field_athlete]
#     athlete_W = features[field_athlete].values

#     K_athlete_pred = gp_utils.transform(
#         lambda: get_athlete_kernel().K(
#             pred_years - data['year0'], data['years'] - data['year0'])
#     ).apply(params)

#     y_athlete_preds = pd.DataFrame(
#         (K_athlete_pred[:, None, :] * athlete_W.T[None, ...]) @ a,
#         index=pred_years,
#         columns=athlete_weights.columns
#     )

#     athletes = data['athlete_results'].groupby("athletes_personId").last()
#     return athletes.join(
#         y_athlete_preds.T,
#         on=field_athlete,
#         how='inner'
#     )
