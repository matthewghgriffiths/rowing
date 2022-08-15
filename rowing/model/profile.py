
from functools import partial 
import numpy as np 

from scipy import optimize 

def periodic(v, box=1):
    return v - np.round(v / box) * box

def logistic(x, A, mean, logscale, box=1):
    # diff = np.c_[mean - box, mean, mean + box] - x[:, None]
    diff = periodic(mean - x, box)
    s = np.exp(logscale)
    diff /= s
    expx = np.exp(diff)
    return A * 4 * (expx / (1 + expx)**2)

def logistics(x, *params, box=1):
    return sum(
        logistic(x, *args, box=box) for args in zip(*[iter(params)]*3)
    )

def observe_logistic(params, t, box):
    return logistics(t, *params, box=box)

def logistic_1(x, A, mean, logscale, box=1, x0=None):
    diff = mean - x
    shift = np.round(diff / box)
    diff -= shift * box
    s = np.exp(logscale)
    x0 = x[0] if x0 is None else x0
    d0 = mean - x0
    i0 = np.round(d0 / box)
    d0 -= i0 * box

    A4s = A * 4 * s
    inta = A4s / (1 + np.exp(diff/s))
    int0 = A4s / (1 + np.exp(d0/s))
    intr1 = A4s / (1 + np.exp(- box/s/2))
    intr2 = A4s / (1 + np.exp(box/s/2))
    
    return inta - int0 - (shift - i0) * (intr1 - intr2)

def logistics_1(x, *params, box=1):
    return sum(
        logistic_1(x, *args, box=box) for args in zip(*[iter(params)]*3)
    )

def guess_peak_times(stroke, stroke_profile):
    t1 = stroke.stroke_time
    return np.r_[
        0.05, 0.4, 
        (t1 - 0.7) * 0.3 + 0.4, (t1 - 0.7) * 0.6 + 0.4, 
        t1 - 0.3, t1 - 0.1
    ]

def fit_stroke_profile(stroke, acc, p0=None, peak_times=None):
    t1 = stroke.stroke_time
    s = np.s_[stroke.stroke_start:stroke.stroke_end]
    stroke_profile = acc[s].copy()
    stroke_profile.index = stroke_profile.index - stroke.stroke_start
    
    if p0 is None:
        if peak_times is None:
            peak_times = guess_peak_times(stroke, stroke_profile)

        peaks = stroke_profile.index[
            stroke_profile.index.searchsorted(peak_times)
        ]
        p0 = np.c_[
            stroke_profile.loc[peaks], 
            peaks, 
            np.full_like(peaks, -3)
        ]

    popt, pcov = optimize.curve_fit(
        lambda t, *params: logistics(t, *params, box=t1), 
        stroke_profile.index, stroke_profile.values, 
        p0=p0.ravel(),
    )
    return stroke_profile, popt, pcov