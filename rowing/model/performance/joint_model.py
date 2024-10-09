
import datetime
from functools import partial, cached_property
import json
from typing import NamedTuple, List, Optional, Dict, Tuple
# from dataclasses import dataclass


import numpy as np
import pandas as pd
# from scipy import sparse
from scipy import stats, sparse

from sklearn import metrics

import jax
from jax import numpy as jnp, tree_map
from jax.scipy.linalg import solve_triangular
from jax.scipy import linalg

from flax.struct import dataclass

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.autonotebook import tqdm

from rowing.world_rowing import api, utils, fields
from rowing.model.gp import kernels, utils as gp_utils

from rowing.model.performance.competition_model import (
    GetKernel, get_race_kernel, get_athlete_kernel, RaceModel,
    year_to_date
)


Message = Tuple[jax.Array, jax.Array]
Messages = List[Dict[str, Message]]
MessageList = Tuple[List[jax.Array], List[jax.Array]]


@jax.jit
def to_natural(message: Message):
    m, var = message
    if np.ndim(var) == 2:
        prec = jnp.linalg.inv(var)
        return prec @ m, prec
    else:
        finite = jnp.isfinite(m) & (var > 0)
        m0 = jnp.where(finite, m, 0)
        v0 = jnp.where(finite, var, 1)
        return jnp.where(finite, m0 / v0, 0), jnp.where(finite, 1 / v0, 0)


@jax.jit
def from_natural(message):
    eta1, eta2 = message
    if np.ndim(eta2) == 2:
        cov = jnp.linalg.inv(eta2)
        return eta1 @ cov, cov

    var = 1 / eta2
    return eta1 * var, var


def ismessage(x):
    # ideally should be isinstance(x, Tuple[np.ndarray, np.ndarray])
    return isinstance(x, tuple)


def listdictzip(**kwargs):
    return [dict(zip(kwargs, vals)) for vals in zip(*kwargs.values())]


def treestarmap(func, vals, **kwargs):
    return jax.tree_map(lambda *x: func(x), *vals, **kwargs)


treemaptuple = partial(treestarmap, tuple)
treemaplist = partial(treestarmap, list)
treemapcat = partial(treestarmap, np.concatenate)


# def treemapcat(vals):
#     return jax.tree_map(lambda *x: np.concatenate(x), *vals)


# def treemaplist(vals):
#     return jax.tree_map(lambda *x: list(x), *vals)


def make_race_model(comp_results):
    boat_order = comp_results.index
    start_times = comp_results['Race Start']
    times = (start_times - start_times.min()).dt.total_seconds().values
    hours = times / 60 * 60

    weights = {}
    for f in [
        # fields.Day,
        "race_event_competition_venueId",
        fields.race_boatClass,
        fields.BoatType,
    ]:
        weights[f] = comp_results.reset_index().groupby([
            "raceBoats_id", f
        ]).size().unstack(level=1, fill_value=0).loc[boat_order]

    weights['lane'] = (
        comp_results.Lane
        - comp_results.Lane.mean()
    ).to_frame().loc[boat_order]
    Ws = {k: jnp.array(df.values) for k, df in weights.items()}
    grams = {k: W @ W.T for k, W in Ws.items()}

    race_model = RaceModel(
        hours=hours,
        W_boatclass=Ws["Boat Class"],
        W_venue=Ws['race_event_competition_venueId'],
        W_lane=Ws['lane'],
        gram_venue=grams['race_event_competition_venueId'],
        gram_boatclass=grams['Boat Class'],
        gram_lane=grams['lane'],
        metadata={
            "results": comp_results,
            "weights": weights,
        }
    )
    return race_model


@dataclass
class ModelData:
    athletes: pd.DataFrame
    seats: pd.DataFrame
    boats: pd.DataFrame
    competitions: pd.DataFrame
    events: pd.DataFrame

    @property
    def athlete_index(self):
        return self.athletes.index

    @property
    def seat_index(self):
        return self.seats.index

    @property
    def boat_index(self):
        return self.boats.index

    @property
    def competition_index(self):
        return self.competitions.index

    @property
    def events_index(self):
        return self.events.index

    @cached_property
    def seat_weights(self):
        return 1 / self.seats.nrowers

    @cached_property
    def athlete_boat_csc(self):
        w = self.seat_weights
        i = self.athlete_index.get_indexer(self.seats.athletes_personId)
        j = self.boat_index.get_indexer(self.seats.athletes_raceBoatId)
        return sparse.csc_array((w, (i, j))).sorted_indices()

    @cached_property
    def boat_competition_csc(self):
        w = np.ones(len(self.boats))
        i = self.boat_index.get_indexer(self.boats.index)
        j = self.competition_index.get_indexer(
            self.boats.race_event_competition_id)
        return sparse.csc_array((w, (i, j))).sorted_indices()

    @cached_property
    def event_boat_csc(self):
        w = np.ones(len(self.boats))
        j = self.boat_index.get_indexer(self.boats.index)
        i = self.events_index.get_indexer(self.boats['Boat Type'])
        return sparse.csc_array((w, (i, j))).sorted_indices()

    @cached_property
    def athlete_competition_csc(self):
        return (self.athlete_boat_csc @ self.boat_competition_csc).sorted_indices()

    @cached_property
    def competition_athlete_csc(self):
        return self.athlete_competition_csc.T.tocsc().sorted_indices()

    @cached_property
    def athlete_performance_order(self):
        return self.athlete_competition_csc.indices.argsort(kind='stable')

    @cached_property
    def competition_performance_order(self):
        return self.competition_athlete_csc.indices.argsort(kind='stable')

    @cached_property
    def athletecompetition_performance_csc(self):
        ath_comp = self.athlete_competition_csc
        i = np.ravel_multi_index(ath_comp.nonzero(), ath_comp.shape)
        j = np.arange(ath_comp.nnz)
        return sparse.csc_array((np.ones(ath_comp.nnz), (i, j)))

    @cached_property
    def athletecompetition_boat_csc(self):
        seats = self.seats
        i = np.ravel_multi_index((
            self.athlete_index.get_indexer(seats.athletes_personId),
            self.competition_index.get_indexer(seats.race_event_competitionId)
        ), self.athlete_competition_csc.shape)
        j = self.boat_index.get_indexer(seats.athletes_raceBoatId)
        return sparse.csc_array((self.seat_weights, (i, j)))

    @cached_property
    def performance_boat_csc(self):
        return self.athletecompetition_performance_csc.T @ self.athletecompetition_boat_csc

    @cached_property
    def competition_boats_groups(self):
        return self.boats.groupby("race_event_competitionId")

    @cached_property
    def competition_athlete_indexers(self):
        split_indices = np.split(
            self.athlete_competition_csc.indices, self.athlete_competition_csc.indptr[1:-1])
        return dict(zip(self.competition_index, split_indices))

    @cached_property
    def competition_athlete_weights(self):
        ath_boat = self.athlete_boat_csc
        ath_inds = self.competition_athlete_indexers
        boat_index = self.boat_index
        return {
            comp: ath_boat[np.ix_(
                ath_inds[comp],
                boat_index.get_indexer_for(comp_results.index)
            )].toarray()
            for comp, comp_results in self.competition_boats_groups
        }

    @cached_property
    def competition_event_weights(self):
        event_boat = self.event_boat_csc
        boat_indexer = self.boat_index.get_indexer
        return {
            comp: event_boat[
                :, boat_indexer(comp_results.index)].toarray()
            for comp, comp_results in self.competition_boats_groups
        }

    @cached_property
    def competition_pgmts(self):
        return {
            comp: comp_results.PGMT.values
            for comp, comp_results in self.competition_boats_groups
        }

    @cached_property
    def competition_models(self):
        return {
            comp: make_race_model(comp_results)
            for comp, comp_results in self.competition_boats_groups
        }

    def split_competition_performance(self, vals):
        return np.split(vals, self.athlete_competition_csc.indptr[1:-1])

    def split_athlete_performance(self, vals):
        return np.split(vals, self.competition_athlete_csc.indptr[1:-1])

    def athlete2competition(self, ath_vals):
        vals = np.concatenate(ath_vals)[self.competition_performance_order]
        return self.split_competition_performance(vals)

    def competition2athlete(self, comp_vals):
        vals = np.concatenate(comp_vals)[self.athlete_performance_order]
        return self.split_athlete_performance(vals)

    def order_competitions(self, vals):
        return [vals[comp] for comp in self.competition_index]


class JointModel:
    def transform(self, dists: List[Message]):
        return dists

    def inv_transform(self, dists: List[Message]):
        return dists

    # def update(self, dists: List[Message], params=None):
    #     # site_dists =


class EventModel(JointModel):
    def calc_posterior(self, dists: List[Message], params=None):
        event_nats = jax.tree_map(to_natural, dists, is_leaf=ismessage)
        event_nat = jax.tree_util.tree_reduce(
            partial(jax.tree_map, jnp.add), event_nats, is_leaf=ismessage)
        return [from_natural(event_nat)] * len(dists), {}


@dataclass
class AthleteModel(JointModel):
    years: np.ndarray
    athlete_year_inds: List[np.ndarray]

    athlete_splits: np.ndarray
    athlete_order: np.ndarray

    competition_splits: np.ndarray
    competition_order: np.ndarray

    athlete_kernel: GetKernel = get_athlete_kernel

    @classmethod
    def from_data(cls, data, **kwargs):
        comp_end = data.competitions["Competition End Date"]
        years = (comp_end.dt.year + comp_end.dt.day_of_year / 365.25).values
        athete_year_inds = np.split(
            data.competition_athlete_csc.indices,
            data.competition_athlete_csc.indptr[1:-1])
        return cls(
            years, athete_year_inds,
            data.athlete_competition_csc.indptr[1:-1],
            data.athlete_competition_csc.indices.argsort(kind='stable'),
            data.competition_athlete_csc.indptr[1:-1],
            data.competition_athlete_csc.indices.argsort(kind='stable'),
            **kwargs
        )

    def apply(self, params, x0=None, x1=None):
        return gp_utils.transform(
            lambda x0, x1: self.athlete_kernel().K(x0, x1)
        ).apply(
            params,
            self.years if x0 is None else x0,
            self.years if x1 is None else x1,
        )

    def split_competition_performance(self, vals):
        return np.split(vals, self.athlete_splits)

    def split_athlete_performance(self, vals):
        return np.split(vals, self.competition_splits)

    def athlete2competition(self, ath_vals):
        vals = np.concatenate(ath_vals)[self.competition_order]
        return self.split_competition_performance(vals)

    def competition2athlete(self, comp_vals):
        vals = np.concatenate(comp_vals)[self.athlete_order]
        return self.split_athlete_performance(vals)

    def ath2comp_message(self, ath_messages: List[Message]) -> List[Message]:
        return treemaptuple(map(
            self.athlete2competition, treemaplist(ath_messages)))

    def comp2ath_message(self, comp_messages: List[Message]) -> List[Message]:
        return treemaptuple(map(
            self.competition2athlete, treemaplist(comp_messages)))

    transform = comp2ath_message
    inv_transform = ath2comp_message

    def calc_posterior(self, dists: List[Message], K_athlete=None, params=None):
        if K_athlete is None:
            K_athlete = self.apply(params)

        post, res = calc_athletes_posterior(
            K_athlete, self.athlete_year_inds, dists)
        return post, res

    def predict(self, dists, times, params):
        K_athlete = self.apply(params)
        K_pred = self.apply(params, times)
        K_00 = self.apply(params, times, times)

        preds = jax.tree_map(
            predict_athlete_scores,
            [K_athlete] * len(dists),
            [K_pred] * len(dists),
            self.athlete_year_inds,
            dists
        )
        return tree_map(
            lambda m: (m[0], K_00 - m[1].T @ m[1]),
            preds,
            is_leaf=ismessage
        )


@jax.jit
def predict_athlete_scores(
    K_athlete: jax.Array, K_pred: jax.Array, i: jax.Array, ath_dist: Message
) -> Tuple[Message, Dict[str, jax.Array]]:

    y, covar = mean_covar(ath_dist)

    K_ath = K_athlete[jnp.ix_(i, i)]
    L = jnp.linalg.cholesky(K_ath + covar)
    Ly = solve_triangular(L, y, lower=True, trans=0)
    a = solve_triangular(L, Ly, lower=True, trans=1)

    k_pr = K_pred[:, i]
    y_pred = (K_pred[:, i] @ a)
    LK = gp_utils.solve_triangular(L, k_pr.T, lower=True, trans=0)

    return (y_pred, LK)


@jax.jit
def calc_athlete_posterior(K_athlete: jax.Array, i: jax.Array, ath_dist: Message) -> Tuple[Message, Dict[str, jax.Array]]:
    y, covar = mean_covar(ath_dist)

    K_ath = K_athlete[jnp.ix_(i, i)]
    L = jnp.linalg.cholesky(K_ath + covar)
    Ly = solve_triangular(L, y, lower=True, trans=0)
    a = solve_triangular(L, Ly, lower=True, trans=1)

    y_post = (K_ath @ a)

    chi2 = Ly.dot(Ly)
    log_marg = - jnp.log(L.diagonal()).sum() - chi2/2

    LK = gp_utils.solve_triangular(L, K_ath, lower=True, trans=0)
    covar_post = K_ath - LK.T @ LK

    return (y_post, covar_post), {'log_marg': log_marg, 'chi2': chi2}


def calc_athletes_posterior(K_athlete, athlete_inds, athlete_dists):
    ret = jax.tree_map(
        calc_athlete_posterior,
        [K_athlete] * len(athlete_inds), athlete_inds, athlete_dists,
    )
    post, res = map(list, zip(*ret))

    return post, res


@dataclass
class CompetitionModels:
    competition_boat_results: List[jax.Array]
    competition_race_models: List[RaceModel]
    competition_weights: List[Dict[str, jax.Array]]

    def apply(self, params):
        return jax.tree_map(
            lambda model: gp_utils.transform(
                model.get_jitter_kernel).apply(params),
            self.competition_race_models,
            is_leaf=lambda x: isinstance(x, RaceModel)
        )

    @classmethod
    def from_data(cls, data):
        race_models = data.order_competitions(data.competition_models)
        pgmts = data.order_competitions(data.competition_pgmts)
        weights = listdictzip(
            athlete=data.order_competitions(data.competition_athlete_weights),
            events=data.order_competitions(data.competition_event_weights),
        )
        return cls(pgmts, race_models, weights)

    def calc_posterior(self, messages, comp_kernels=None, params=None):
        if comp_kernels is None:
            comp_kernels = self.apply(params)

        ret = jax.tree_map(
            calc_comp_athlete_posterior,
            self.competition_boat_results,
            comp_kernels,
            self.competition_weights,
            messages
        )
        post, res = map(list, zip(*ret))
        return post, res

    def calc_update(self, comp_kernels, messages):
        post, res = self.calc_posterior(comp_kernels, messages)
        site_dists = update_site_distribution(post, messages)
        return site_dists, post, res


@jax.jit
def calc_comp_athlete_posterior(y_boat, K_boat, weights, cavity_dists):
    K_race = K_boat
    y_race = y_boat
    K_cavity = {}
    for k, Wk in weights.items():
        y_k, var_k = cavity_dists[k]
        if var_k.ndim == 1:
            K_k = (Wk.T * var_k).T
            Kkk = Wk.T @ K_k
            var_k = jnp.diag(var_k)
        else:
            K_k = var_k @ Wk
            Kkk = K_k.T @ jsp.linalg.solve(var_k, Wk, assume_a='pos')

        K_race += Kkk
        y_race -= y_k @ Wk
        K_cavity[k] = K_k, Wk, y_k, var_k

    L = jnp.linalg.cholesky(K_race)
    Ly = gp_utils.solve_triangular(L, y_race, lower=True, trans=0)
    a = gp_utils.solve_triangular(L, Ly, lower=True, trans=1)

    chi2 = Ly.dot(Ly)
    log_marg = - jnp.log(L.diagonal()).sum() - chi2/2

    post_dists = {}
    for k, (K_k, Wk, y_k, var_k) in K_cavity.items():
        y_post = K_k @ a + y_k
        LKab = gp_utils.solve_triangular(L, K_k.T, lower=True, trans=0)
        covar_post = jnp.diag(var_k) - LKab.T @ LKab
        post_dists[k] = y_post, covar_post

    res = {'log_marg': log_marg, 'chi2': chi2}
    return post_dists, res


def mean_var(cav):
    m, var = cav
    if jnp.ndim(var) == 2:
        var = var.diagonal()
    return m, var


def mean_covar(cav):
    m, covar = cav
    if jnp.ndim(covar) == 1:
        covar = jnp.diag(covar)

    return m, covar


@jax.jit
def div_normal_var(cav1, cav2):
    # m1, var1 = mean_var(cav1)
    # m2, var2 = mean_var(cav2)

    # var = 1 / (1/var1 - 1/var2)
    # m = (m1 / var1 - m2 / var2) * var
    # return m, var
    cav1, cav2 = to_natural(mean_var(cav1)), to_natural(mean_var(cav2))
    return from_natural(jax.tree_map(jnp.subtract, cav1, cav2))


def mul_normal_var(cav1, cav2):
    # m1, var1 = mean_var(cav1)
    # m2, var2 = mean_var(cav2)
    # prec1 = 1 / var1
    # prec2 = 1 / var2
    # var = 1 / (prec1 + prec2)
    # m = (prec1 * m1 + prec2 * m2) * var
    # return m, var
    cav1, cav2 = to_natural(mean_var(cav1)), to_natural(mean_var(cav2))
    return from_natural(jax.tree_map(jnp.add, cav1, cav2))


def update_site_distribution(posterior: Messages, cavity: Messages) -> Messages:
    return jax.tree_map(
        div_normal_var,
        posterior, cavity,
        is_leaf=ismessage
    )  # List[Dict[str, Message]]


def update_site_messages(posterior: Messages, cavity: Messages) -> Messages:
    site_dists = update_site_distribution(
        posterior, cavity)  # List[Dict[str, Message]]

    # Dict[str, List[Message]]
    return treemaplist(site_dists, is_leaf=ismessage)


def apply_normal_prior(prior, dists: List[Message]):
    if prior:
        return tree_map(partial(mul_normal_var, prior), dists, is_leaf=ismessage)
    else:
        return dists


def predict_performances(times, athlete_model, athlete_dists, data, params):
    ath_preds = athlete_model.predict(athlete_dists, times, params)

    athlete_preds = pd.concat({
        "score": pd.concat({
            ath: pd.Series(pred[0], times)
            for ath, pred in zip(data.athlete_index, ath_preds)
        }, names=['athlete_id']),
        "score_std": pd.concat({
            ath: pd.Series(pred[0], times)
            for ath, pred in zip(data.athlete_index, ath_preds)
        }, names=['athlete_id']),
    }, axis=1).rename_axis(
        index=['athlete_id', 'year']
    ).reset_index().join(
        data.athletes[[
            'athletes_person_BirthDate', 'athletes_person'
        ]],
        on='athlete_id'
    )
    athlete_preds['Date'] = year_to_date(athlete_preds.year).dt.normalize()

    return athlete_preds


def combine_performances(athlete_dists, data):
    performance_post = np.concatenate([
        m for m, _ in athlete_dists])
    performance_std = np.concatenate([
        np.sqrt(cov.diagonal()) for _, cov in athlete_dists])

    return pd.DataFrame({
        "score": performance_post,
        "score_std": performance_std,
        "athlete_id": data.athlete_index[
            data.athlete_competition_csc.nonzero()[0]
        ],
        "competition_id": data.competition_index[
            data.athlete_competition_csc.nonzero()[1]
        ]
    })


def load_competition(competition_id):
    competition = api.get_worldrowing_record("competition", competition_id)

    events = api.get_events(
        competition_id, include="boats.boatAthletes.person"
    ).join(
        boat_types, on='event_boatClassId'
    )

    boat_cols = ['Event', "Boat Type", 'n_rowers']

    boats = pd.json_normalize(sum(events.event_boats, [])).join(
        events.set_index("event_id")[boat_cols], on='eventId'
    )
    boats = boats[boats.boatAthletes.str[0].notna()]
    boats['Crew Names'] = boats['DisplayName'] + " " + boats['Boat Type']
    boats['Country'] = boats['DisplayName'].str[:3]
    boats['Final Position'] = boats.finalRank
