
import numpy as np 
from functools import cached_property

from filterpy.kalman import (
    UnscentedKalmanFilter as UKF, 
    MerweScaledSigmaPoints as MSSP, 
    JulierSigmaPoints as JSP,
)

def weight_mean(points, weights):
    return np.dot(weights, points)

def _calc_cov(diff1, diff2, weights):
    return np.einsum("i,ij,ik->jk", weights, diff1, diff2)

def weight_mean_cov(points, weights, points2=None):
    mean1 = weight_mean(points, weights)
    diff1 = points - mean1[None, :]
    if points2 is None:
        points2 = points
        mean2, diff2 = mean1, diff1
    else:
        mean2 = weight_mean(points2, weights)
        diff2 = points2 - mean2[None, :]

    cov = _calc_cov(diff1, diff2, weights)
    return mean1, mean2, cov    


class SigmaPoints:
    def __init__(self, points, unscented_transform):
        self.points = points 
        self.UT = unscented_transform
        self.Wm = self.UT.Wm 
        self.Wc = self.UT.Wc 
    
    @cached_property
    def mean(self) -> np.ndarray:
        return self.Wm.dot(self.points)

    @cached_property 
    def diff(self) -> np.ndarray:
        return self.points - self.mean[None, :]

    @cached_property
    def cov(self) -> np.ndarray:
        return self.cross_cov(self)

    def cross_cov(self, other: "SigmaPoints") -> np.ndarray:
        return np.einsum("i,ij,ik->jk", self.Wc, self.diff, other.diff)

    def from_points(self, new_points):
        return type(self)(new_points, self.UT)

    def from_mean_cov(self, m, cov):
        return self.from_points(self.UT.sigma_points(m, cov))

    def __iter__(self):
        return iter(self.points)

    @property 
    def shape(self):
        return self.points.shape 

    @property 
    def size(self):
        return self.points.size

    def __getitem__(self, index):
        return self.points[index]

class Kalman:
    def __init__(self, transfer, noise=0, noise_fun=None):
        self.transfer = transfer 
        self.noise_fun = noise_fun 
        self.noise = noise

    def apply_to_points(self, points, *args, **kwargs):
        return points.from_points(np.array([
            self.transfer(p, *args, **kwargs) for p in points
        ]))

    def add_noise(self, cov, *args, **kwargs):
        if self.noise_fun:
            noise = self.noise + self.noise_fun(*args, **kwargs)
        else:
            noise = self.noise

        if np.ndim(noise) < 2:
            cov.flat[::cov.shape[0] + 1] += noise 
        else:
            cov += noise

        return cov


class UnscentedKalmanTransition(Kalman):    
    def __init__(self, transfer=lambda x: x, noise=0, noise_fun=None):
        super().__init__(transfer, noise=noise, noise_fun=noise_fun)

    def unscented_update(self, points: SigmaPoints, *args, **kwargs):
        new_points = self.apply_to_points(points, *args, **kwargs)
        new_points.cov = self.add_noise(new_points.cov, *args, **kwargs)
        return new_points 


class UnscentedKalmanObserve(Kalman):
    def unscented_observe(self, observation, points: SigmaPoints, *args, **kwargs):
        pred_obs = self.apply_to_points(points, *args, **kwargs)
        
        innovation = observation - pred_obs.mean 
        innovation_cov = self.add_noise(pred_obs.cov, *args, **kwargs)

        cov_pred_points = pred_obs.cross_cov(points)
        kalman_gain = np.linalg.solve(innovation_cov, cov_pred_points)

        new_mean = points.mean + innovation.dot(kalman_gain) 
        new_cov = points.cov + kalman_gain.T.dot(innovation_cov.dot(kalman_gain))

        return points.from_mean_cov(new_mean, new_cov)