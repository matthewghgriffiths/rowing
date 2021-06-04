

from collections import namedtuple
from functools import cached_property

import numpy as np 
import pandas as pd
from scipy import linalg, integrate, stats

from . import api, utils, livetracker
from .livetracker import RaceTracker


def calc_win_probs(times, std):
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


def load_predicter(noise=10., data_path=utils._data_path):
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
        mean_cov + cov_pace, 
        index=distances, 
        columns=distances 
    )
    return PredictRace(distances, K, noise=noise)


class PredictRace:
    def __init__(self, distances, K, noise=10.):
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
        # cached inverses
        self.pace_predicters = {}
        self.pred_pace_covs = {}

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
            live_dist_data[('pace', cnt)] = 500 / live_dist_data.metrePerSecond[cnt]
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
        self.pace_predicters[distance] = pace_predicter
        
        LK = linalg.solve_triangular(
            LXX, kxX.T, lower=True
        )
        pred_pace_cov = pd.DataFrame(
            kxx - LK.T.dot(LK), index=x, columns=x
        )
        self.pred_pace_covs[distance] = pred_pace_cov
        
        return pace_predicter, pred_pace_cov
        
    def get_linear_pace_predicter(self, distance):
        if distance not in self.pace_predicters:
            self.calc_predicters(distance)
            
        return self.pace_predicters[distance]
    
    def get_pred_pace_cov(self, distance):
        if distance not in self.pred_pace_covs:
            self.calc_predicters(distance)
            
        return self.pred_pace_covs[distance]

    def predict(self, race_pace):
        pred_pace = race_pace.copy()
        pred_times = race_pace.copy()
        pred_distance = race_pace.copy()
        pred_pace_cov = {}
        pred_times_cov = {}
        pred_distance_cov = {}

        M = self.pace_time_M
        for cnt, cnt_pace in race_pace.items():
            distance = cnt_pace.index[cnt_pace.notna()][-1]
            pace_predictor = self.get_linear_pace_predicter(distance)
            pred_pace[cnt] = pace_predictor.dot(cnt_pace.loc[:distance])
            pred_pace_cov[cnt] = self.get_pred_pace_cov(distance)
            pred_times[cnt] = M.dot(pred_pace[cnt])
            pred_times_cov[cnt] = M.dot(pred_pace_cov[cnt].dot(M.T))

        leader_time = pred_times.min(1)
        for cnt in pred_times.columns:
            maxi = pred_times[cnt].searchsorted(leader_time.iloc[-1]) + 1
            pred_distance[cnt] = np.interp(
                leader_time, 
                pred_times[cnt].iloc[:maxi], 
                pred_times.index[:maxi], 
            )
            pred_speed = 500 / pred_pace[cnt].values[:, None]
            pred_distance_cov[cnt] = pred_times_cov[cnt] * pred_speed * pred_speed.T

        pred_pace_std, pred_times_std, pred_distance_std = (
            pd.DataFrame(
                {
                    cnt: np.sqrt(cov.values.diagonal())
                    for cnt, cov in pred_pace_cov.items()
                }, 
                index=race_pace.index
            )
            for pred_cov in (
                pred_pace_cov, pred_times_cov, pred_distance_cov, 
            )
        )
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
            pace_predictor = self.get_linear_pace_predicter(distance)
            pred_pace = pace_predictor.dot(race_pace.loc[:distance])
            pred_pace_cov = self.get_pred_pace_cov(distance)
            return pred_pace, pred_pace_cov
        else:
            pred_pace = {}
            pred_pace_cov = {}
            for cnt, cnt_pace in race_pace.items():
                distance = cnt_pace.index[cnt_pace.notna()][-1]
                pace_predictor = self.get_linear_pace_predicter(distance)
                pred_pace[cnt] = pace_predictor.dot(cnt_pace.loc[:distance])
                pred_pace_cov[cnt] = self.get_pred_pace_cov(distance)
                
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
        noise=10.,
        data_path=utils._data_path,
        **kwargs
    ):
        super().__init__(race_id, **kwargs)
        self.predicter = \
            predicter or load_predicter(noise=noise, data_path=data_path)
        
        self.pace_preds = {}
        self.time_preds = {}
        self.finish_preds = {}
        self.win_preds = {}

    def update_livedata(self):
        super().update_livedata()
        self.race_pace = self.predicter.calc_boat_pace(
            self.live_data
        )
        self.race_times = self.predicter.calc_boat_times(
            self.live_data
        )
        return self.live_data

    def predict_pace(self, distance=None, update=False):
        if distance and distance in self.pace_preds:
            return self.pace_preds[distance]

        if update:
            self.update_livedata()

        preds = self.predicter.predict_pace(
            self.race_pace, distance=distance
            )
        distance = distance or self.race_pace.index[-1]
        self.pace_preds[distance] = preds
        return preds

    def predict_times(self, distance=None, update=False):
        if distance and distance in self.time_preds:
            return self.time_preds[distance]

        if update:
            self.update_livedata()

        pace_preds, time_preds = self.predicter.predict_pace_times(
            self.race_pace, distance=distance)
        distance = distance or self.race_pace.index[-1]
        self.time_preds[distance] = time_preds
        self.pace_preds[distance] = pace_preds
        return time_preds

    def predict_finish_time(self, distance=None, update=False):
        if distance and distance in self.finish_preds:
            return self.finish_preds[distance]

        if update:
            self.update_livedata()

        preds = self.predicter.predict_finish_time(
            self.race_pace, distance=distance)
        distance = distance or self.race_pace.index[-1]
        self.finish_preds[distance] = preds
        return preds

    def predict_win_probability(self, distance=None, update=False):
        pred_finish, finish_std = self.predict_finish_time(
            update=update, distance=distance
        )
        win_probs = calc_win_probs(pred_finish, finish_std)
        distance = distance or self.race_pace.index[-1]
        self.win_preds[distance] = win_probs
        return win_probs







def calc_boat_time(live_data, distances=None):
    distances = distances or np.linspace(0, 2000, 401).astype(int)
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
        ),
        index=dist_travelled.columns,
        columns=distances
    )
    boat_times[0] = 0
    boat_times.columns.name = 'distance'
    boat_times.index.name = 'country'
    return boat_times

def calc_boat_pace(live_data, distances=None):
    distances = distances or np.linspace(0, 2000, 401).astype(int)
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
        ),
        index=live_data.metrePerSecond.columns,
        columns=distances
    )
    boat_pace.columns.name = 'distance'
    boat_pace.index.name = 'country'
    return boat_pace


def calc_all_boat_pace(race_live_data, distances=None, set_last=True):
    distances = distances or np.linspace(0, 2000, 401).astype(int)
    boat_pace = pd.concat({
        race_id: calc_boat_pace(live_data)
        for race_id, live_data in race_live_data.items()
    }) 
    if set_last:
        boat_pace.loc[:, distances[-1]] = boat_pace.loc[:, distances[-2]]

    return boat_pace

PacePred = namedtuple('PacePred', 'pace, pace_cov, time, time_cov')

def predict_pace_time(pred_distances, pace, K, L=None, noise=0.1):
    x = pred_distances
    X = pace.index
    
    kxx = K.loc[x, x]
    kxX = K.loc[x, X]
    if L is None:
        KXX = K.loc[X, X] + np.eye(len(X)) * noise
        L = pd.DataFrame(
            np.linalg.cholesky(KXX),
            index=KXX.index,
            columns=KXX.columns, 
        ) 
    else:
        KXX = K.loc[X, X]
        L = L.loc[X,X]
    
    pred_pace = kxX.dot(linalg.cho_solve((L, True), pace))
    Kkh = linalg.solve_triangular(
        L, kxX.T, lower=True
    )
    pred_pace_cov = kxx - Kkh.T.dot(Kkh)
    
    deltam = np.diff(pred_distances)
    triu = np.triu(
        np.ones_like(kxx) * np.r_[deltam, deltam[0]]
    )/500
    pred_times = pred_pace.dot(triu)
    pred_times_cov = triu.T.dot(pred_pace_cov).dot(triu)
    
    return PacePred(
        pred_pace, pred_pace_cov, 
        pred_times, pred_times_cov)


def pred_finish_time(pred_distances, pace, K, noise=0.1, start_distance=250):
    x = pred_distances
    X = pace.index
    
    kxx = K.loc[x, x]
    L = pd.DataFrame(
        np.linalg.cholesky(
            K.loc[X, X] + np.eye(len(X)) * noise
        ),
        index=K.index,
        columns=K.columns, 
    )

    n = X.searchsorted(start_distance)
    pred_finish = pace.loc[start_distance:].copy()
    pred_finish_std = pace.iloc[start_distance:].copy()
    pred_finish.name = 'predicted_finish_time'
    pred_finish_std.name = 'predicted_finish_time_std'

    for d in pred_finish.index:
        X = pace.loc[:d].index
        kxX = K.loc[x, X]
        LXX = L.loc[X, X]
        pred_pace = kxX.dot(linalg.cho_solve((LXX, True), pace.loc[:d]))
        Kkh = linalg.solve_triangular(
            LXX, kxX.T, lower=True
        )
        pred_pace_cov = kxx - Kkh.T.dot(Kkh)
        deltam = np.diff(x)    
        deltam = np.r_[deltam, deltam[0]]/500
        pred_finish[d] = pred_pace.dot(deltam)
        pred_finish_std[d] = deltam.dot(pred_pace_cov).dot(deltam)

    return pred_finish, pred_finish_std




def fit_factor_analysis_regularised(X, d, F=None, W=None, delta=1, psi=1, niter=100):
    """
    d: number of factors
    delta: derivative regularisation factor
    psi: Gaussian magnitude regularisation factor
    """
    n, m = X.shape
    #initialise factors
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
    B = delta * D.dot(D.T) # Derivative regularisation
    Fp = F.copy()
    Wp = W.copy()
    for i in range(niter):
        A = Wp.T.dot(Psi.dot(Wp))
        S = A.diagonal()[:, None] +  B.diagonal()[None, :]
        C = 2 * W.T.dot(Psi.dot(X))
        D = (C - A.dot(Fp) - Fp.dot(B))/S/2
        Fp += D
        Wp = np.linalg.lstsq(Fp.T, X.T, rcond=None)[0].T

    return Wp, Fp
