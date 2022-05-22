
import sys
import logging
from typing import Callable, Dict, TypeVar, Tuple, Any
from contextlib import nullcontext
import string 
from datetime import timedelta  
import re 

from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from concurrent.futures import ProcessPoolExecutor
except ModuleNotFoundError:
    pass

import numpy as np


_YMD = "%Y-%m-%d"

_LOGGING_LEVELS = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG,
}

def add_logging_argument(parser):
    parser.add_argument(
        "-l"
        "-log", 
        "--log", 
        '--logging',
        default="warning",
        help=(
            "Provide logging level. "
            "Example --log debug', default='warning'"
        ),
    )


def set_logging(options):
    level = _LOGGING_LEVELS.get(options.log.lower())
    if level is None:
        raise ValueError(
            f"log level given: {options.log}"
            f" -- must be one of: {' | '.join(_LOGGING_LEVELS.keys())}")

    logging.basicConfig(level=level)


def random_alphanumeric(size, p=None):
    alphanumeric = string.ascii_letters + string.digits
    return ''.join(np.random.choice(list(alphanumeric), size=40))


_MSH_STR_FORMAT = "{minutes:d}:{seconds:02d}.{hundredths:02d}"
_HMSH_STR_FORMAT = "{hours}:{minutes:02d}:{seconds:02d}.{hundredths:02d}"
def strfsplit(tdelta, hours=False):
    components = tdelta.components._asdict()
    components['hundredths'] = tdelta.components.milliseconds // 10
    if tdelta.components.hours or hours:
        return _HMSH_STR_FORMAT.format(**components)
    else:
        return _MSH_STR_FORMAT.format(**components)


def distance_to_km(d):
    dist, km = re.match(r"([0-9\.]+)\s*(k?)m", d).groups()
    return float(dist) * (1 if km else 1e-3)

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


def unflatten_json(entity, key=()):
    if isinstance(entity, dict):
        for k, val in entity.items():
            yield from unflatten_json(val, key + (k,))
    elif isinstance(entity, list):
        for i, elem in enumerate(entity):
            yield from unflatten_json(elem, key + (i,))
    else:
        yield key, entity

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

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


def flatten_json(obj, key=''):
    if isinstance(obj, dict):
        key = key + '_' if key else ''
        for k, val in obj.items():
            yield from flatten_json(val, key + k)
    elif isinstance(obj, list):
        key = key + '_' if key else ''
        for i, val in enumerate(obj):
            yield from flatten_json(val, key + str(i))
    else:
        yield key, obj 