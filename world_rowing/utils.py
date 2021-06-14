
import os
from pathlib import Path
from functools import lru_cache
from datetime import timedelta, datetime
import logging
from typing import Callable, Dict, TypeVar, Tuple, Any
from contextlib import nullcontext

from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from concurrent.futures import ProcessPoolExecutor
except ModuleNotFoundError:
    ProcessPoolExecutor = None

import numpy as np
import pandas as pd
from scipy.special import erf

_file_path = Path(os.path.abspath(__file__))
_module_path = _file_path.parent
_data_path = _module_path / 'data'
_flag_path = _data_path / 'flags'

# Make lru_cache 3.7 compatible
cache = lru_cache(maxsize=128)


def Phi(z):
    sq2 = 1.4142135623730951
    return 1/2 + erf(z/sq2)/2


CURRENT_TIMEZONE = datetime.now().astimezone().tzinfo


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


def format_totalseconds(seconds, hundreths=True):
    return format_timedelta(timedelta(seconds=seconds), hundreths=hundreths)


def format_timedelta(td, hours=False, hundreths=True):
    mins, secs = divmod(td.seconds, 60)
    if hours:
        hs, mins = divmod(mins, 60)
        return f"{hs:02d}:{mins:02d}:{secs:02d}"
    else:
        end = f".{(td.microseconds // 10_000):02d}" if hundreths else ''
        return f"{mins}:{secs:02d}{end}"


def format_timedelta_hours(td):
    return format_timedelta(td, hours=True)


def format_yaxis_splits(ax=None, ticks=True, hundreths=False):
    format_axis_splits(ax=ax, yticks=ticks, hundreths=hundreths)


def format_xaxis_splits(ax=None, ticks=True, hundreths=False):
    format_axis_splits(ax=ax, yticks=False, xticks=ticks, hundreths=hundreths)


def format_axis_splits(ax=None, yticks=True, xticks=False, hundreths=False):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    if yticks:
        if yticks is True:
            yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels(
            [format_totalseconds(s, hundreths) for s in yticks]
        )
    if xticks:
        if xticks is True:
            xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [format_totalseconds(s, hundreths) for s in xticks]
        )


K = TypeVar("K")
A = TypeVar('A')
V = TypeVar('V')


def map_concurrent(
    func: Callable[..., V],
    inputs: Dict[K, Tuple],
    threaded: bool = True,
    max_workers: int = 10,
    show_progress: bool = True,
    raise_on_err: bool = False,
    **kwargs,
) -> Tuple[Dict[K, V], Dict[K, Exception]]:
    """
    This function is equalivant to calling,

    >>> output = {k: func(*args, **kwargs) for k, args in inputs.items()}

    except that the function is called using either `ThreadPoolExecutor` 
    if `threaded=True` or a `ProcessPoolExecutor` otherwise.

    The function returns a tuple of `(output, errors)` where errors returns
    the errors that happened during the calling of any of the functions. So
    the function will run all the other work before 

    The function also generates a status bar indicating the progress of the
    computation.

    Alternatively if `raise_on_err=True` then the function will reraise the
    same error.

    Examples
    --------
    >>> import time
    >>> def do_work(arg):
    ...     time.sleep(0.5)
    ...     return arg
    >>> inputs = {i: (i,) for i in range(20)}
    >>> output, errors = map_concurrent(do_work, inputs)
    100%|███████████████████| 20/20 [00:01<00:00, 19.85it/s, completed=18]
    >>> len(output), len(errors)
    (20, 0)

    >>> def do_work2(arg):
    ...     time.sleep(0.5)
    ...     if arg == 5:
    ...         raise(ValueError('something went wrong'))
    ...     return arg
    >>> output, errors = map_concurrent(do_work2, inputs)
    100%|████████| 20/20 [00:01<00:00, 19.86it/s, completed=18, nerrors=1]
    >>> len(output), len(errors)
    (19, 1)
    >>> errors
{5: ValueError('something went wrong')}

    >>> try:
    ...     output, errors = map_concurrent(
    ...         do_work2, inputs, raise_on_err=True)
    ... except ValueError:
    ...     print("task failed successfully!")
    ...
    45%|█████████▍           | 9/20 [00:00<00:00, 17.71it/s, completed=5]
    task failed!
    """
    output = {}
    errors = {}

    Executor = ThreadPoolExecutor if threaded else ProcessPoolExecutor

    if show_progress:
        from tqdm.auto import tqdm
        pbar = tqdm(total=len(inputs))
    else:
        pbar = nullcontext()
    with Executor(max_workers=max_workers) as executor, pbar:
        work = {
            executor.submit(func, *args, **kwargs): k
            for k, args in inputs.items()
        }
        status: Dict[str, Any] = {}
        for future in as_completed(work):
            status['completed'] = key = work[future]
            if show_progress:
                pbar.update(1)
                pbar.set_postfix(**status)
            try:
                output[key] = future.result()
            except Exception as exc:
                if raise_on_err:
                    raise exc
                else:
                    logging.warning(f"{key} experienced error {exc}")
                    errors[key] = exc
                    status['nerrors'] = len(errors)

    return output, errors


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
