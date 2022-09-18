
import sys 
import logging 
from typing import Callable, Dict, TypeVar, Tuple, Any, Optional 
from contextlib import nullcontext
from datetime import timedelta, datetime
import re 
import json 
from functools import lru_cache
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from concurrent.futures import ProcessPoolExecutor
except ModuleNotFoundError:
    ProcessPoolExecutor = None

from tqdm.auto import tqdm 
import pandas as pd
import numpy as np

# Python 3.7 compatible cached_property 
try:
    from functools import cached_property 
except ImportError:
    from ._compat import cached_property 

_pyodide = "pyodide" in sys.modules

logger = logging.getLogger(__name__)

@lru_cache
def load_gsheet(sheet):
    from gspread_pandas import Spread
    return Spread(sheet)

def to_gspread(
    df, spread, sheet_name="Sheet1", 
    index=True, header=True, merge_cells=True, freeze=True, 
    max_workers=4, **kwargs
):
    spread = load_gsheet(spread)
    n_index = int(index)
    n_header = int(header)

    spread.freeze(0, 0, sheet_name)

    if merge_cells:
        flat_df = df 
        rowstart = df.columns.nlevels + 1
        if index: 
            flat_df = flat_df.reset_index()
        if header:
            flat_df = flat_df.T.reset_index().T

        spread.unmerge_cells(sheet=sheet_name)
        spread.df_to_sheet(
            flat_df, index=False, headers=False, **kwargs
        )
        n_index = n_index and df.index.nlevels
        n_header = n_header and df.columns.nlevels
        merge_groups = []
        if index:
            merge_groups += [
                (
                    (merge_group['row'], merge_group['col']),
                    (merge_group['mergestart'], merge_group['mergeend']),
                ) for merge_group in _to_merge_index(df.index, rowstart, 1)
                if merge_group['mergeend'] and merge_group['mergestart']
            ]
        if header:
            merge_groups += [
                (
                    (merge_group['col'], merge_group['row']),
                    (merge_group['mergeend'], merge_group['mergestart']),
                ) for merge_group in _to_merge_index(df.columns, n_index + 1, 1)
                if merge_group['mergeend'] and merge_group['mergestart']
            ]

        if merge_groups:
            map_concurrent(
                spread.merge_cells, merge_groups, sheet=sheet_name, max_workers=max_workers)
    else:
        spread.df_to_sheet(
            df, index=index, headers=header, sheet=sheet_name, **kwargs
        )

    if freeze:
        spread.freeze(
            cols=n_index,
            rows=n_header,
            sheet=sheet_name
        )

    sheet = spread.find_sheet(sheet_name)

    if kwargs.get("replace"):
        sheet.format(
            to_a1_notation(1, 1, *spread.get_sheet_dims(sheet_name)),
            {
                "verticalAlignment": "BOTTOM",
                "horizontalAlignment": "LEFT",
                "textFormat": {
                    "bold": False
                }
            }
        )
    sheet.format(
        to_a1_notation(1, 1, n_header + len(df), n_index),
        {
            "verticalAlignment": "TOP",
            "textFormat": {
                "bold": True
            }
        }
    )
    sheet.format(
        to_a1_notation(1, 1, n_header, df.shape[1] + n_index),
        {
            "horizontalAlignment": "CENTER",
            "textFormat": {
                "bold": True
            }
        }
    )
    return spread

def read_gspread(sheet, sheet_name="Sheet1", **kwargs):
    gspread = load_gsheet(sheet)
    return gspread.sheet_to_df(sheet=sheet_name, **kwargs)

def _to_merge_index(index, rowcounter=0, gcolidx=0):
    import pandas as pd 
    if not isinstance(index, pd.MultiIndex):
        index = pd.MultiIndex.from_product([index])

    level_strs = index.format(
        sparsify=True, adjoin=False, names=False
    )
    level_lengths = pd.io.formats.format.get_level_lengths(
        level_strs
    )

    for lnum, (spans, levels, level_codes) in enumerate(
        zip(level_lengths, index.levels, index.codes)
    ):
        values = levels.take(
            level_codes,
            allow_fill=levels._can_hold_na,
            fill_value=levels._na_value,
        )
        for i, span_val in spans.items():
            mergestart, mergeend = None, None
            if span_val > 1:
                mergestart = rowcounter + i + span_val - 1
                mergeend = gcolidx

            yield dict(
                row=rowcounter + i,
                col=gcolidx,
                val=values[i],
                mergestart=mergestart,
                mergeend=mergeend,
            )
        gcolidx += 1


def to_a1_notation(*args):
    import gspread 
    
    if len(args) == 4:
        range_start = gspread.utils.rowcol_to_a1(*args[:2])
        range_end = gspread.utils.rowcol_to_a1(*args[-2:])
        range_name = ":".join((range_start, range_end))
        return range_name 
    elif len(args) == 2:
        return gspread.utils.rowcol_to_a1(*args[:2])
    else:
        raise ValueError("incorrect number of arguments passed")


def to_timestamp(time, unit="1s"):
    return (pd.to_datetime("today") - pd.Timestamp(0)) // pd.Timedelta(unit)


def from_timestamp(timestamp, epoch=pd.Timestamp(0), unit="1s"):
    return epoch + pd.to_timedelta(timestamp, unit=unit)


def format_totalseconds(seconds, hundreths=True):
    return format_timedelta(timedelta(seconds=seconds), hundreths=hundreths)


def format_timedelta(td, hours=False, hundreths=True):
    mins, secs = divmod(td.seconds, 60)
    secs = int(secs)
    end = f".{(td.microseconds // 10_000):02d}" if hundreths else ''
    if hours:
        hs, mins = divmod(mins, 60)
        return f"{hs:02d}:{mins:02d}:{secs:02d}{end}"
    else:
        return f"{mins}:{secs:02d}{end}"


def format_timedelta_hours(td, hundreths=True):
    return format_timedelta(td, hours=True, hundreths=hundreths)


def format_series_timedelta(s):
    na_vals = s.isna()
    components = s.fillna(pd.Timedelta(0)).dt.components
    components['hundredths'] = components.milliseconds // 10
    components = components.astype(str)
    times = (
        components.hours + ":" 
        + components.minutes.str.zfill(2) 
        + ":" + components.seconds.str.zfill(2) 
        + "." + components.hundredths.str.zfill(2) 
    )
    times[na_vals] = ''
    return times 


def format_gsheet(df, index=True, columns=True):
    gsheet = df.copy()
    for c, col in gsheet.items():
        col_types = col.apply(type).unique()
        if np.isin(pd.Timedelta, col_types):
            gsheet[c] = format_series_timedelta(col)

    if index:
        gsheet.index = pd.MultiIndex.from_frame(format_gsheet(
            df.index.to_frame(), index=False, columns=False
        ))
        
    if columns:
        gsheet.columns = pd.MultiIndex.from_frame(format_gsheet(
            df.columns.to_frame(), index=False, columns=False
        ))

    return gsheet


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
    progress_bar=tqdm,
    singleton: bool = False, 
    total: Optional[int] = None, 
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

    if isinstance(inputs, dict):
        items = inputs.items()
    elif isinstance(inputs, list):
        items = enumerate(inputs)
        total = len(inputs)
        if not total:
            return [], {}
    else:
        items = inputs

    if singleton:
        def get(args):
            return args, 
    else:
        def get(args):
            return args 
        
    pbar = progress_bar(total=total or len(items)) if progress_bar else nullcontext()
    with pbar, Executor(max_workers=max_workers) as executor:
        work = {executor.submit(func, *get(args), **kwargs): k for k, args in items}

        status: Dict[str, Any] = {}
        for future in as_completed(work):
            status["completed"] = key = work[future]
            if progress_bar:
                pbar.update(1)
                pbar.set_postfix(**status)
            try:
                output[key] = future.result()
                # pylint: disable=broad-except
            except Exception as exc:
                if raise_on_err:
                    raise exc
                else:
                    logging.warning("%s experienced error %s", key, exc)
                    errors[key] = exc
                    status["nerrors"] = len(errors)

    if isinstance(inputs, list):
        output = [output.get(i, None) for i in range(len(inputs))]

    return output, errors


def _map_singlethreaded(
    func: Callable[..., V],
    inputs: Dict[K, Tuple],
    threaded: bool = True,
    max_workers: int = 10,
    progress_bar=tqdm,
    raise_on_err: bool = False,
    **kwargs,
) -> Tuple[Dict[K, V], Dict[K, Exception]]:
    output = {}
    errors = {}

    status: Dict[str, Any] = {}    
    pbar = progress_bar(len(inputs)) if progress_bar else nullcontext()
    with pbar:
        for key, args in inputs.items():
            try:
                output[key] = func(*args, **kwargs)
            except Exception as exc:
                if raise_on_err:
                    raise exc
                else:
                    logging.warning(f"{key} experienced error {exc}")
                    errors[key] = exc
                    status['nerrors'] = len(errors)

            if show_progress:
                pbar.update(1)
                pbar.set_postfix(**status)

    return output, errors


if _pyodide:
    map_concurrent = _map_singlethreaded


def cached_map_concurrent(
    func: Callable[..., V],
    inputs: Dict[K, Tuple],
    singleton: bool = False, 
    local_cache: Optional["LocalCache"] = None, 
    path = ".", 
    total = None,
    **kwargs,
):
    if local_cache is None:
        logger.debug("no cache for %r", func)
        return map_concurrent(
            func, inputs, **kwargs
        )

    if isinstance(inputs, dict):
        items = inputs.items()
    elif isinstance(inputs, list):
        items = enumerate(inputs)
    else:
        items = inputs

    total = total or len(items)

    if singleton:
        def get(args):
            return args, 
    else:
        def get(args):
            return args 

    path = Path(path)
    def cached_func(k, *args, **kwargs):
        return local_cache.get(k, path, func, *args, **kwargs)

    cached_inputs = ((k, (k, *get(args))) for k, args in items)

    output, errors = map_concurrent(
        cached_func, cached_inputs, total=total, **kwargs
    )

    if isinstance(inputs, list):
        output = [output.get(i, None) for i in range(len(inputs))]

    return output, errors


class LocalCache:
    def __init__(
            self, serialise, deserialise, file_ending, 
            read_mode=None, write_mode=None
        ):
        self._serialise = serialise 
        self._deserialise = deserialise
        self.file_ending = file_ending
        self.read_mode = read_mode 
        self.write_mode = write_mode

    def get_path(self, key, path):
        obj_path = Path(path)
        if isinstance(key, tuple):
            for k in key[:-1]:
                obj_path = obj_path / str(k)
            key = key[-1]
            
        obj_path = obj_path / f"{key}.{self.file_ending}"
        return obj_path

    def get(self, key, path, func, *args, **kwargs):
        obj_path = self.get_path(key, path)
        if obj_path.is_file():
            logger.debug("deserialising %r at %s", key, obj_path)
            return self.deserialise(obj_path)
        else:
            return self.update(key, path, func, *args, **kwargs)

    def update(self, key, path, func, *args, **kwargs):
        obj_path = self.get_path(key, path)
        obj = func(*args, **kwargs)
        logger.debug("serialising %r at %s", key, obj_path)
        self.serialise(obj, obj_path)
        return obj

    def serialise(self, obj, obj_path, **kwargs):
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        if self.write_mode:
            with open(obj_path, self.write_mode) as f:
                self._serialise(obj, f, **kwargs)
        else:
            self._serialise(obj, obj_path, **kwargs)

        return obj_path 

    def deserialise(self, obj_path, **kwargs):
        if self.read_mode:
            with open(obj_path, self.read_mode) as f: 
                return self._deserialise(f, **kwargs)
        else:
            return self._deserialise(obj_path, **kwargs)


parquet_cache = LocalCache(
    pd.DataFrame.to_parquet, 
    pd.read_parquet, 
    "parquet"
)

json_cache = LocalCache(
    json.dump, json.load, "json", "r", "w"
)


class CachedClient:
    def __init__(
            self, username=None, password=None, path=None, 
            local_cache: Optional[LocalCache] = None, map_kws = None
        ):
        self.username = username 
        self.password = password 
        self.path = Path(path).resolve()
        self.local_cache = local_cache
        self.map_kws = map_kws or {}

    @classmethod 
    def from_credentials(cls, credentials, **kwargs):
        if not isinstance(credentials, dict):
            with open(credentials, 'r') as f:
                credentials = json.load(f)

        return cls(**credentials, **kwargs)

    def map_concurrent(self, func, *args, **kwargs):
        return map_concurrent(func, *args, **{**self.map_kws, **kwargs})

    def cached(self, key, func, *arg, local_cache=None, path=None, reload=False, **kwargs):
        local_cache = local_cache or self.local_cache 
        path = path or self.path 
        if local_cache:
            if reload:
                return local_cache.update(
                    key, path, func, *arg, **kwargs
                )
            else:
                return local_cache.get(
                    key, path, func, *arg, **kwargs
                )
        else:
            func(*args, **kwargs)