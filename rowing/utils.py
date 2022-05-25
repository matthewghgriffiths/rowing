
import sys 
import logging 
from typing import Callable, Dict, TypeVar, Tuple, Any
from contextlib import nullcontext
from datetime import timedelta, datetime

from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from concurrent.futures import ProcessPoolExecutor
except ModuleNotFoundError:
    ProcessPoolExecutor = None

# Python 3.7 compatible cached_property 
try:
    from functools import cached_property 
except ImportError:
    from ._compat import cached_property 

_pyodide = "pyodide" in sys.modules


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


def _map_singlethreaded(
    func: Callable[..., V],
    inputs: Dict[K, Tuple],
    threaded: bool = True,
    max_workers: int = 10,
    show_progress: bool = True,
    raise_on_err: bool = False,
    **kwargs,
) -> Tuple[Dict[K, V], Dict[K, Exception]]:
    output = {}
    errors = {}

    status: Dict[str, Any] = {}
    if show_progress:
        from tqdm.auto import tqdm
        pbar = tqdm(total=len(inputs))
    else:
        pbar = nullcontext()

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