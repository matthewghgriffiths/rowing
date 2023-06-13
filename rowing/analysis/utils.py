
import logging
import string
from datetime import timedelta
import re
import numpy as np
from typing import Optional

from ..utils import (
    cached_property, map_concurrent, format_totalseconds,
    format_timedelta, format_timedelta_hours,
    format_yaxis_splits, format_xaxis_splits, format_axis_splits,
    _to_merge_index, load_gsheet, to_gspread, cached_map_concurrent,
    json_cache, parquet_cache, format_series_timedelta, format_gsheet,
    CachedClient, to_timestamp, from_timestamp
)
from .. import utils

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


def add_credentials_arguments(parser):
    parser.add_argument(
        '-u', '--user', '--email',
        type=str, nargs='?',
        help='Email address to use'
    )
    parser.add_argument(
        '-p', '--password',
        type=str, nargs='?',
        help='Password'
    )
    parser.add_argument(
        '-c', '--credentials',
        type=str, nargs='?',
        help='path to json file containing credentials (email and password)'
    )


def add_gspread_arguments(parser):
    parser.add_argument(
        '--gspread',
        type=str, nargs='?',
        help='name, url, or id of the spreadsheet'
    )


def load_gspread(options):
    if options.gspread:
        return load_gsheet(options.gspread)


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


def unflatten_json(entity, key=()):
    if isinstance(entity, dict):
        for k, val in entity.items():
            yield from unflatten_json(val, key + (k,))
    elif isinstance(entity, list):
        for i, elem in enumerate(entity):
            yield from unflatten_json(elem, key + (i,))
    else:
        yield key, entity


def is_pareto_efficient(costs, return_mask=True):
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
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(
            costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # Remove dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1

    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


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
