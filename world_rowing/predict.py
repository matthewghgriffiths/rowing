

from collections import namedtuple
from functools import cached_property

import numpy as np 
import pandas as pd
from scipy import linalg, integrate, stats

from . import api, utils, livetracker

def calc_win_probs(times, std):
    diffs = times.min() - times.values 

    def func(x):
        logcdf = -stats.norm.logcdf((x - diffs), loc=std)
        logcdf -= logcdf.sum()
        logp = stats.norm.logpdf((x - diffs), loc=std)
        return np.exp(logp + logcdf)

    win_prob, err = integrate.quad_vec(
        func, -np.inf, np.inf, 
    )
    win_prob /= win_prob.sum()
    return pd.Series(win_prob, index=times.index)


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
    
    def predict_pace(self, race_pace, distance=None):
        distance = distance or race_pace.index[-1]
        pace_predictor = self.get_linear_pace_predicter(distance)
        pred_pace = pace_predictor.dot(race_pace.loc[:distance])
        pred_pace_cov = self.get_pred_pace_cov(distance)
        return pred_pace, pred_pace_cov
    
    def predict_time(
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
 
        

class RaceTracker:
    def __init__(
            self, race_id,
            predicter=None,
            gmt=None,  
            colors=None, 
            live_data=None, 
            results=None, 
            intermediates=None,
    ):
        self.race_id = race_id
        
        self.predicter = predicter
        self.gmt = gmt or api.find_world_best_time(
            race_id=race_id
        ).ResultTime.total_seconds()   
        
        self.colors = \
            colors or plt.rcParams['axes.prop_cycle'].by_key()['color']
        
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
        return api.get_worldrowing_record('race', self.race_id)
    
    @cached_property
    def race_boats(self):
        return api.get_race_results(race_id=self.race_id).reset_index()
    
    @cached_property
    def countries(self):
        return self.race_boats.Country

    @property
    def country_colors(self):
        return dict(zip(self.countries, self.colors))
    
    @cached_property
    def event_id(self):
        return self.race_details.eventId
    
    @cached_property
    def event_details(self):
        return api.get_worldrowing_record('event', self.event_id)
    
    @cached_property
    def competition_id(self):
        return self.event_details.competitionId
    
    @cached_property
    def competition_details(self):
        return api.get_worldrowing_record(
            'competition', self.competition_id)
    
    def update_livedata(self):
        self.live_data, self.results, self.intermediates = \
            livetracker.get_race_livetracker(
                self.race_id, 
                gmt=self.gmt,
                race_distance=self.race_distance,
                cached=False,
        )
        
    def plot(self, distance, y, ax=None, maxdistance=None, **kwargs):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        lines = {}
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
    
    def plot_pace(self, ax=None, **kwargs):
        distance = self.live_data.distanceTravelled
        pace = 500 / self.live_data.metrePerSecond
        ax, lines = self.plot(distance, pace, ax=ax, **kwargs)
        utils.format_yaxis_splits(ax)
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
