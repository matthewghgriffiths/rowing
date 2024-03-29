
import numpy as np
import pandas as pd
from scipy import integrate


def integrate_series(s, initial=0, **kwargs):
    return pd.Series(
        integrate.cumtrapz(s, s.index, initial=initial),
        index=s.index, **kwargs
    )


def interpolate_series(index, s, **kwargs):
    return pd.Series(
        np.interp(index, s.index, s.values, **kwargs),
        index=index
    )
