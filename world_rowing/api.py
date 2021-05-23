

from functools import lru_cache
import datetime 
from collections.abc import Mapping

import pandas as pd
import requests

from .utils import getnesteditem, map_concurrent

def stringify_value(value):
    if isinstance(value, (str, int, float)):
        return str(value)
    else:
        return "||".join(map(str, value))
    

def prepare_options(key, options):
    _options = options.items() if isinstance(options, Mapping) else options
    return (
        (f"{key}[{field}]", stringify_value(value))
        for field, value in _options
    )

def prepare_params(**kwargs):
    return {
        field: value
        for key, options in kwargs.items()
        for field, value in prepare_options(key, options)
    }


def get_worldrowing_data(*endpoints, **kwargs):
    endpoint = "/".join(endpoints)
    url = f"https://world-rowing-api.soticcloud.net/stats/api/{endpoint}"
    params = prepare_params(**kwargs) or None
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()['data']


def get_worldrowing_records(*endpoints, **kwargs):
    records =  pd.DataFrame.from_records(
        get_worldrowing_data(*endpoints, **kwargs)
    )
    for col in records.columns:
        if 'Date' in col:
            dates = pd.to_datetime(records[col])
            if dates.notna().all():
                records[col] = dates
    
    return records



cached_worldrowing_data = lru_cache(get_worldrowing_data)
cached_worldrowing_records = lru_cache(get_worldrowing_records)

@lru_cache
def get_competitions(year=None, fisa=True, **kwargs):
    year = year or datetime.date.today().year
    kwargs.setdefault('filter', {})['Year'] = year
    kwargs.setdefault('filter', {})['IsFisa'] = 1 if fisa else ''
    kwargs.setdefault('sort', {'StartDate': 'asc'})
    return get_worldrowing_records(
        'competition', 
        **kwargs
    )

def get_this_years_competitions(fisa=True):
    year = datetime.date.today().year
    return get_competitions(year=year, fisa=fisa)

def get_most_recent_competition(fisa=True):
    competitions = get_this_years_competitions(fisa=True)
    started = competitions.StartDate < datetime.datetime.now()
    return competitions.loc[started].iloc[-1]

def get_boat_types():
    return cached_worldrowing_records('boatClass')

def get_competition_types():
    return cached_worldrowing_records('competitionType')
    
def get_statistics():
    return cached_worldrowing_records('statistic')
    
def get_venues():
    return cached_worldrowing_records('venue')
    

WBT_RECORDS = {
    'Date': ('DateOfBT',), 
    'BoatClass': ('BoatClass',),
    'Venue': ('Venue', 'Name'),
    'Competition': ('Competition', 'Name'),
    'Race': ('Race', 'Name'),
    'Country': ('Competitor', 'Nationality', 'Name'),
}

def _extract_wbt_record(record):
    return {
        key: getnesteditem(record, *items)
        for key, items in WBT_RECORDS.items()
    }

@lru_cache
def load_world_best_times(json_url):
    return pd.DataFrame.from_records(
        map(
            _extract_wbt_record, 
            requests.get(json_url).json()['BestTimes']
        )
    )
    
@lru_cache
def get_world_best_times():
    wbt_stats = get_worldrowing_records(
        'statistic', 
        filter={'Category': 'WBT'}
    )
    wbts, _ = map_concurrent(
        load_world_best_times, 
        dict(zip(wbt_stats.description, zip(wbt_stats.url))),
        show_progress=False, 
    )
    wbts = pd.concat(wbts, names=['Description'])\
        .reset_index(0)\
        .reset_index(drop=True)
    wbts.Description = wbts.Description.str.extract(
        r": ([a-zA-Z]+)"
    )
    return wbts