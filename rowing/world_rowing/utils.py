
import os
import sys
from pathlib import Path
from functools import lru_cache, wraps
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
from scipy.special import erf

from ..utils import (
    cached_property, map_concurrent, format_totalseconds,
    format_timedelta, format_timedelta_hours, 
    format_yaxis_splits, format_xaxis_splits, format_axis_splits
)

_pyodide = "pyodide" in sys.modules


_file_path = Path(os.path.abspath(__file__))
_module_path = _file_path.parent
_data_path = _module_path.parent.parent / 'data'
_flag_path = _data_path / 'flags'

# Make lru_cache 3.7 compatible
cache = lru_cache(maxsize=128)


def Phi(z):
    sq2 = 1.4142135623730951
    return 1/2 + erf(z/sq2)/2


CURRENT_TIMEZONE = datetime.now().astimezone().tzinfo


def ignore_nans(func):
    @wraps(func)
    def nan_func(arg, *args, **kwargs):
        if np.isnan(arg):
            return arg
        else:
            return func(arg, *args, **kwargs)

    return nan_func


def first(it, *args):
    return next(iter(it), *args)


def read_times(times):
    minfmt = ~ times.fillna('').str.match(r"[0-9]+:[0-9][0-9]?:[0-9][0-9]?")
    new_times = times.copy()
    new_times[minfmt] = "0:" + times
    return pd.to_timedelta(new_times)


@cache
def get_iso_country_data(data_path=_data_path / 'iso_country.json'):
    return pd.read_json(data_path)


@cache
def get_iso2_names(data_path=_data_path / 'iso_country.json'):
    iso_country = get_iso_country_data(data_path)
    return pd.concat([
        pd.Series(
            iso_country.iso2[col.notna()].values,
            index=col[col.notna()],
        )
        for _, col in iso_country.iteritems()
    ])


@cache
def get_iso3_names(data_path=_data_path / 'iso_country.json'):
    iso_country = get_iso_country_data(data_path)
    return pd.concat([
        pd.Series(
            iso_country.iso3[col.notna()].values,
            index=col[col.notna()].str.upper(),
        )
        for _, col in iso_country.iteritems()
    ])


@cache
def find_country_iso2(
    country,
    data_path=_data_path / 'iso_country.json'
):
    return get_iso2_names(data_path)[country]


@cache
def get_flag_im(
        country,
        data_path=_data_path / 'iso_country.json',
        flag_path=_flag_path
):
    import matplotlib.pyplot as plt
    iso2cnt = find_country_iso2(
        country, data_path
    )
    im = plt.imread(
        flag_path / f"{iso2cnt}.png"
    )
    return im


def update_table(table, index, columns, update):
    index, columns = map(np.asarray, (index, columns))
    for row, rowvals in update.iterrows():
        i, = np.where(index == row)[0]
        for col, val in rowvals.items():
            j, = np.where(columns == col)[0]
            table[i + 1, j].get_text().set_text(val)


def make_flag_box(
        country,
        xy,
        base_width=900,
        zoom=0.04,
        resample=True,
        frameon=False,
        xycoords='data',
        box_alignment=(0.5, 0.),
        pad=0,
        data_path=_data_path / 'iso_country.json',
        flag_path=_flag_path,
        offset_kws=None,
        **kwargs
):
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    im = get_flag_im(
        country,
        data_path=data_path,
        flag_path=flag_path,
    )
    offset_im = OffsetImage(
        im,
        zoom=zoom * base_width / im.shape[1],
        resample=resample,
        **(offset_kws or {})
    )
    return AnnotationBbox(
        offset_im, xy,
        frameon=frameon,
        xycoords=xycoords,
        box_alignment=box_alignment,
        pad=0,
        **kwargs
    )


def update_fill_between(poly, x, y0, y1):
    x, y0, y1 = map(np.asarray, (x, y0, y1))
    vertices = poly.get_paths()[0].vertices
    vertices[1:len(x)+1, 0] = x
    vertices[1:len(x)+1, 1] = y0
    vertices[len(x) + 1:-2, 0] = x[::-1]
    vertices[len(x) + 1:-2, 1] = y1[::-1]
    vertices[0, 0] = vertices[-1, 0] = x[0]
    vertices[0, 1] = vertices[-1, 1] = y1[0]


def update_fill_betweenx(poly, y, x0, x1):
    y, x0, x1 = map(np.asarray, (y, x0, x1))
    vertices = poly.get_paths()[0].vertices
    n = len(y)
    vertices[1:n+1, 0] = x0
    vertices[1:n+1, 1] = y
    vertices[n + 1:-2, 0] = x1[::-1]
    vertices[n + 1:-2, 1] = y[::-1]
    vertices[0, 0] = vertices[-1, 0] = x1[0]
    vertices[0, 1] = vertices[-1, 1] = y[0]


def getnesteditem(container, *items):
    value = container
    for item in items:
        value = value[item]

    return value


def extract_fields(record, fields):
    return {
        k: getnesteditem(record, *items) for k, items in fields.items()
    }


def merge(dfs, **kwargs):
    import pandas as pd
    left = dfs[0]
    n = len(dfs[1:])

    def iterate(val):
        if isinstance(val, tuple):
            return iter(val)
        else:
            return (val for _ in range(n))

    iter_kwargs = {k: iterate(val) for k, val in kwargs.items()}

    for right in dfs[1:]:
        kws = {
            k: next(val) for k, val in iter_kwargs.items()
        }
        left = pd.merge(left, right, **kws)

    return left
