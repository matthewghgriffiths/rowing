

from typing import Dict
from collections import namedtuple
from functools import cached_property
import logging

import numpy as np
import pandas as pd
from scipy import linalg, integrate, stats

from . import api, utils, livetracker
from .livetracker import RaceTracker
from .utils import cache, lru_cache


def calc_win_probs(times, std):
    if np.allclose(std, 0):
        win_prob = np.zeros_like(times)
        win_prob[times.argmin()] = 1
        return pd.Series(win_prob, index=times.index)

    norm = stats.norm(
        loc=times.min() - times.values,
        scale=np.asarray(std)
    )

    def func(x):
        logcdf = -norm.logcdf(x)
        logcdf -= logcdf.sum()
        logp = norm.logpdf(x)
        return np.exp(logp + logcdf)

    win_prob, err = integrate.quad_vec(
        func, -np.inf, np.inf,
    )
    win_prob /= win_prob.sum()
    return pd.Series(win_prob, index=times.index)


@cache
def load_predicter(noise=1., data_path=utils._data_path):
    mean_pace = pd.read_csv(
        data_path / 'mean_pace.csv.gz'
    ).set_index('distance').pace
    distances = mean_pace.index

    cov_pace = pd.read_csv(
        data_path / 'cov_pace.csv.gz',
        index_col=0
    )
    cov_pace.columns = cov_pace.columns.astype(int)
    assert (
        (cov_pace.index == distances).all() and
        (cov_pace.columns == distances).all()
    )
    mean_cov = mean_pace.values[:, None] * mean_pace.values[None, :]
    K = pd.DataFrame(
        mean_cov + 0.5*cov_pace,
        index=distances,
        columns=distances
    )
    return PredictRace(distances, K, noise=noise)


class PredictRace:
    def __init__(self, distances, K, noise=1.):
        self.distances = pd.Index(distances, name='distance')
        # Distance travelled between each distance
        # Needed to calculate times
        deltam = np.diff(self.distances)
        deltam = np.r_[deltam, deltam[0]]
        self.delta_dist = pd.Series(
            deltam, index=self.distances
        )
        # covariance matrix
        self.K = pd.DataFrame(
            K,
            index=self.distances,
            columns=self.distances
        )
        # Cholesky factorisation of covariance matrix with noise
        self.L = pd.DataFrame(
            np.linalg.cholesky(
                K + np.eye(len(K)) * noise
            ),
            index=K.index,
            columns=K.columns,
        )
        # linear transform to calculate predicted times
        self.pace_time_M = pd.DataFrame(
            np.tril(
                np.ones_like(K)
                * (self.delta_dist.values[:, None]/500)
            ),
            index=self.distances,
            columns=self.distances
        )

    def calc_distance_data(self, live_data):
        distances = self.distances
        columns = [
            'distanceFromLeader', 'strokeRate',
            'metrePerSecond', 'PGMT'
        ]
        dist_travelled = live_data.distanceTravelled
        boat_lims = {
            cnt: dist_travelled[cnt].searchsorted(distances[-1]) + 1
            for cnt in dist_travelled.columns
        }
        parse_cols = live_data[columns].columns
        live_dist_data = pd.DataFrame(
            np.vstack(
                [
                    np.interp(
                        distances,
                        dist_travelled.loc[:boat_lims[cnt], cnt],
                        live_data.loc[:boat_lims[cnt], (col, cnt)],
                        right=np.nan
                    )
                    for (col, cnt) in parse_cols
                ]
            ).T,
            columns=parse_cols,
            index=distances
        )
        for cnt in live_dist_data.metrePerSecond.columns:
            live_dist_data[('pace', cnt)] = 500 / \
                live_dist_data.metrePerSecond[cnt]

        for cnt in live_dist_data.metrePerSecond.columns:
            live_dist_data[('time', cnt)] = np.interp(
                distances,
                dist_travelled.loc[:boat_lims[cnt], cnt],
                live_data.time.loc[:boat_lims[cnt]],
                right=np.nan
            )

        return live_dist_data

    def calc_boat_times(self, live_data):
        distances = self.distances
        dist_travelled = live_data.distanceTravelled
        boat_time_lims = (
            (cnt, dist_travelled[cnt].searchsorted(distances[-1]) + 1)
            for cnt in dist_travelled.columns
        )
        boat_times = pd.DataFrame(
            np.vstack(
                [
                    np.interp(
                        distances,
                        dist_travelled[cnt][:i],
                        live_data.time[:i]
                    )
                    for cnt, i in boat_time_lims
                ]
            ).T,
            columns=dist_travelled.columns,
            index=distances
        )
        boat_times[0] = 0
        return boat_times

    def calc_boat_pace(self, live_data):
        distances = self.distances
        boat_pace = pd.DataFrame(
            np.vstack(
                [
                    np.interp(
                        distances,
                        live_data.distanceTravelled[cnt],
                        500 / live_data.metrePerSecond[cnt]
                    )
                    for cnt in live_data.metrePerSecond.columns
                ]
            ).T,
            columns=live_data.metrePerSecond.columns,
            index=distances
        )
        return boat_pace

    @lru_cache(maxsize=512)
    def calc_predicters(self, distance):
        kxx = self.K.loc[:, :]
        kxX = self.K.loc[:, :distance]
        LXX = self.L.loc[:distance, :distance]
        x = kxx.columns
        X = kxX.columns

        pace_predicter = pd.DataFrame(
            linalg.cho_solve((LXX, True), kxX.T).T,
            index=x, columns=X,
        )
        LK = linalg.solve_triangular(
            LXX, kxX.T, lower=True
        )
        pred_pace_cov = pd.DataFrame(
            kxx - LK.T.dot(LK), index=x, columns=x
        )
        return pace_predicter, pred_pace_cov

    def predict(self, live_data, match_to_live=True):
        live_dist_data = self.calc_distance_data(live_data)
        race_pace = live_dist_data.pace

        pred_pace = race_pace.copy()
        pred_times = race_pace.copy()
        pred_distance = race_pace.copy()
        pred_pace_cov = {}
        pred_times_cov = {}
        pred_distance_cov = {}

        M = self.pace_time_M
        for cnt, cnt_pace in race_pace.items():
            distance = cnt_pace.index[cnt_pace.notna()][-1]
            pace_predictor, pred_pace_cov[cnt] = self.calc_predicters(distance)

            pred_pace[cnt] = pace_predictor.dot(cnt_pace.loc[:distance])
            pred_times[cnt] = M.dot(pred_pace[cnt])
            pred_times_cov[cnt] = M.dot(pred_pace_cov[cnt].dot(M.T))

        for cnt in race_pace.columns:
            pred_time = pred_times[cnt]
            time = live_dist_data.time[cnt]
            diff = pred_time - time
            mean_diff = diff.mean()
            curr_diff = diff.dropna().iloc[[-1]] - mean_diff
            pace_diff = 500 / curr_diff.index[0] * curr_diff.values[0]

            pred_pace[cnt] -= pace_diff
            pred_times[cnt] = M.dot(pred_pace[cnt]) - mean_diff

        leader_time = pred_times.min(1)
        for cnt in pred_times.columns:
            maxi = pred_times[cnt].searchsorted(leader_time.iloc[-1]) + 1
            pred_distance[cnt] = np.interp(
                leader_time,
                pred_times[cnt].iloc[:maxi],
                pred_times.index[:maxi],
            )
            pred_speed = 500 / pred_pace[cnt].values[:, None]
            pred_distance_cov[cnt] = pred_times_cov[cnt] * \
                pred_speed * pred_speed.T

        pred_pace_std, pred_times_std, pred_distance_std = (
            pd.DataFrame(
                {
                    cnt: np.sqrt(cov.values.diagonal())
                    for cnt, cov in pred_pace_cov.items()
                },
                index=race_pace.index
            )
            for pred_cov in (
                pred_pace_cov, pred_times_cov, pred_distance_cov
            )
        )
        pred_distance_std = pred_times_std * (500 / pred_pace)

        if match_to_live:
            for cnt in race_pace.columns:
                pace = live_dist_data.pace[cnt]
                pred_pace.loc[pace.notna(), cnt] = pace.dropna()
                pred_pace_std.loc[pace.notna(), cnt] = 0
                time = live_dist_data.time[cnt]
                pred_times.loc[time.notna(), cnt] = time.dropna()
                pred_times_std.loc[time.notna(), cnt] = 0
                dist = (
                    live_dist_data.index
                    - live_dist_data.distanceFromLeader[cnt])
                pred_distance.loc[dist.notna(), cnt] = dist.dropna()
                pred_distance_std.loc[dist.notna(), cnt] = 0

        win_probs = calc_win_probs(
            pred_times.iloc[-1],
            pred_times_std.iloc[-1]
        )

        return (
            (pred_pace, pred_pace_std),
            (pred_times, pred_times_std),
            (pred_distance, pred_distance_std),
            win_probs
        )

    def predict_pace(self, race_pace, distance=None):
        if distance:
            pace_predictor, pred_pace_cov = self.calc_predicters(distance)
            pred_pace = pace_predictor.dot(race_pace.loc[:distance])
            pred_pace_cov = self.get_pred_pace_cov(distance)
            return pred_pace, pred_pace_cov
        else:
            pred_pace = {}
            pred_pace_cov = {}
            for cnt, cnt_pace in race_pace.items():
                distance = cnt_pace.index[cnt_pace.notna()][-1]
                pace_predictor, pred_pace_cov[cnt] = self.calc_predicters(
                    distance)
                pred_pace[cnt] = pace_predictor.dot(cnt_pace.loc[:distance])

            return pd.DataFrame(pred_pace), pd.concat(pred_pace_cov, axis=1)

    def predict_pace_times(self, race_pace, distance=None):
        pred_pace, pred_pace_cov = self.predict_pace(
            race_pace, distance)

        M = self.pace_time_M
        pred_times = M.dot(pred_pace)
        if distance:
            pred_time_cov = M.dot(pred_pace_cov.dot(M.T))
        else:
            pred_time_cov = pd.concat({
                cnt: M.dot(pred_pace_cov[cnt].dot(M.T))
                for cnt in pred_times.columns
            }).T

        return (pred_pace, pred_pace_cov), (pred_times, pred_time_cov)

    def predict_times(
            self, race_pace, distance=None,
            pred_pace=None, pred_pace_cov=None
    ):
        if pred_pace is None or pred_pace_cov is None:
            pred_pace, pred_pace_cov = self.predict_pace(
                race_pace, distance)

        M = self.pace_time_M
        pred_times = M.dot(pred_pace)
        pred_time_cov = M.dot(pred_pace_cov.dot(M.T))

        return pred_times, pred_time_cov

    def predict_finish_time(
            self, race_pace, distance=None,
            pred_pace=None, pred_pace_cov=None
    ):
        if pred_pace is None or pred_pace_cov is None:
            pred_pace, pred_pace_cov = self.predict_pace(
                race_pace, distance)

        v = self.pace_time_M.iloc[-1]
        pred_finish = v.dot(pred_pace)
        pred_finish_cov = v.dot(pred_pace_cov.dot(v))
        return pred_finish, pred_finish_cov


class LivePrediction(RaceTracker):
    def __init__(
        self, race_id,
        predicter: PredictRace = None,
        noise=0.3,
        data_path=utils._data_path,
        **kwargs
    ):
        super().__init__(race_id, **kwargs)
        self.predicter = \
            predicter or load_predicter(noise=noise, data_path=data_path)

    def predict(self, live_data=None, match_to_live=True):
        if live_data is None:
            if self.live_data is None:
                live_data = self.update_livedata()
            else:
                live_data = self.live_data

        if len(live_data):
            return self.predicter.predict(live_data, match_to_live=match_to_live)


def fit_factor_analysis_regularised(X, d, F=None, W=None, delta=1, psi=1, niter=100):
    """
    d: number of factors
    delta: derivative regularisation factor
    psi: Gaussian magnitude regularisation factor
    """
    n, m = X.shape
    # initialise factors
    if F is None:
        U, s, Vt = np.linalg.svd(X)
        scale = np.diag(np.sqrt(s[:d]))
        W = U[:, :d].dot(scale)
        F = scale.dot(Vt[:d, :])
    if W is None:
        W = np.linalg.lstsq(F.T, X.T, rcond=None)[0].T

    D = np.zeros((m, m-1))
    ind = np.diag_indices(m-1)
    D[ind] = - 1
    D[ind[0] + 1, ind[1]] = 1

    # Do Expectation Maximisation
    Psi = psi * np.eye(n)  # Gaussian regularisation
    B = delta * D.dot(D.T)  # Derivative regularisation
    Fp = F.copy()
    Wp = W.copy()
    for i in range(niter):
        A = Wp.T.dot(Psi.dot(Wp))
        S = A.diagonal()[:, None] + B.diagonal()[None, :]
        C = 2 * W.T.dot(Psi.dot(X))
        D = (C - A.dot(Fp) - Fp.dot(B))/S/2
        Fp += D
        Wp = np.linalg.lstsq(Fp.T, X.T, rcond=None)[0].T

    return Wp, Fp
