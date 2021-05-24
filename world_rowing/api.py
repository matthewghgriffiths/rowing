

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
    if isinstance(options, str):
        return (key, options), 

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

def request_worldrowing_data(*endpoints, **kwargs):
    endpoint = "/".join(endpoints)
    url = f"https://world-rowing-api.soticcloud.net/stats/api/{endpoint}"
    params = prepare_params(**kwargs) or None
    return requests.get(url, params=params)

cached_request_worldrowing_data = lru_cache(request_worldrowing_data)


def get_worldrowing_data(*endpoints, cached=True, **kwargs):
    if cached:
        hashable_kws = {
            k: vals if isinstance(vals, (str, int, float))
            else tuple(dict(vals).items())
            for k, vals in kwargs.items()
        }
        r = cached_request_worldrowing_data(*endpoints, **hashable_kws)
    else:
        r = request_worldrowing_data(*endpoints, **kwargs)
    r.raise_for_status()
    if r.text:
        return r.json()['data']
    else:
        return []


def get_worldrowing_record(*endpoints, **kwargs):
    data = get_worldrowing_data(*endpoints, **kwargs)
    return pd.Series(data, name=data.pop('id'))


def get_worldrowing_records(*endpoints, **kwargs):
    records =  pd.DataFrame.from_records(
        get_worldrowing_data(*endpoints, **kwargs)
    )
    if 'id' in records.columns:
        records.set_index('id', inplace=True)

    for col in records.columns:
        if 'Date' in col:
            dates = pd.to_datetime(records[col])
            if dates.notna().all():
                records[col] = dates
    
    return records


def get_competition_events(competition_id, cached=True):
    return get_worldrowing_records(
        'event', 
        cached=True,
        filter=(
            ('competitionId', competition_id),
        ), 
        sort=(
            ('eventId', 'asc'),
            ('Date', 'asc')
        )
    )

def get_competition_races(competition_id, cached=True):
    return get_worldrowing_records(
        'race', 
        cached=cached,
        filter=(
            ('event.competitionId', competition_id),
        ), 
        sort=(
            ('eventId', 'asc'),
            ('Date', 'asc')
        )
    )

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
    return get_worldrowing_records('boatClass', cached=True)

def get_competition_types():
    return get_worldrowing_records('competitionType', cached=True)
    
def get_statistics():
    return get_worldrowing_records('statistic', cached=True)
    
def get_venues():
    return get_worldrowing_records('venue', cached=True)
    
RACEBOAT_FIELDS = {
    'id': ('id',),
    'raceId': ('raceId',),
    'boatId': ('boatId',),
    'Country': ('DisplayName',),
    'Rank': ('Rank',),
    'Lane': ('Lane',),
    'ResultTime': ('ResultTime',),
}
def _extract(record, fields):
    return {
        k: getnesteditem(record, *items) for k, items in fields.items()
    }

def get_race_results(
        race_id=None, 
        event_id=None, 
        competition_id=None, 
        cached=True, 
        **kwargs
        ):
    filters = tuple(dict(kwargs.pop('filter', ())).items())
    if race_id:
        filters += ('id', race_id),
    if event_id:
        filters += ('eventId', event_id),
    if competition_id:
        filters += ('event.competitionId', competition_id),

    race_data = get_worldrowing_data(
        'race', 
        cached=cached, 
        filter=filters, 
        include='raceBoats.raceBoatIntermediates',
        **kwargs
    )
    results = pd.DataFrame.from_records([
        _extract(boat, RACEBOAT_FIELDS)
        for race in race_data for boat in race['raceBoats']
        if boat['ResultTime']
    ])
    if len(results):
        results.set_index(['raceId', 'id'], inplace=True)
        results.ResultTime = pd.to_timedelta(results.ResultTime)
    return results

RACEINTERMEDIATE_FIELDS = {
    'id': ('id',),
    # 'raceId': 'raceId',
    'boatId': ('raceBoatId',),
    'Rank': ('Rank',),
    'ResultTime': ('ResultTime',),
    'distanceId': ('distanceId',),
    'distance': ('distance', 'DisplayName'),
}

def get_intermediate_results(
        race_id=None, 
        event_id=None, 
        competition_id=None, 
        cached=True, 
        **kwargs
        ):
    filters = tuple(dict(kwargs.pop('filter', ())).items())
    if race_id:
        filters += ('id', race_id),
    if event_id:
        filters += ('eventId', event_id),
    if competition_id:
        filters += ('event.competitionId', competition_id),
        
    race_data = get_worldrowing_data(
        'race', 
        cached=cached, 
        filter=filters, 
        include='raceBoats.raceBoatIntermediates.distance',
        **kwargs
    )
    race_fields = RACEBOAT_FIELDS.copy()
    race_fields['raceBoatId'] = race_fields.pop('id')
    results = pd.DataFrame.from_records([
        {
            **_extract(boat, race_fields), 
            **_extract(inter, RACEINTERMEDIATE_FIELDS)
        }
        for race in race_data 
        for boat in race['raceBoats'] if boat['ResultTime']
        for inter in boat['raceBoatIntermediates']
    ])
    
    if len(results):
        results.set_index(['raceId', 'raceBoatId', 'id'], inplace=True)
        results.ResultTime = pd.to_timedelta(results.ResultTime)
    return results



WBT_RECORDS = {
    'BoatClass': ('BoatClass',),
    'ResultTime': ('Competitor', 'ResultTime'),
    'Competition': ('Competition', 'Name'),
    'Venue': ('Venue', 'Name'),
    'Event': ('Event', 'DisplayName'),
    'EventId': ('Event', 'Id'),
    'Race': ('Race', 'Name'),
    'RaceId': ('Race', 'Id'),
    'Country': ('Competitor', 'Nationality', 'Name'),
    'Date': ('DateOfBT',), 
}

def _extract_wbt_record(record):
    return {
        key: getnesteditem(record, *items)
        for key, items in WBT_RECORDS.items()
    }


@lru_cache
def load_competition_best_times(json_url):
    return pd.DataFrame.from_records(
        _extract(record, WBT_RECORDS)
        for record in requests.get(json_url).json()['BestTimes']
    )
    

@lru_cache
def get_competition_best_times():
    wbt_stats = get_worldrowing_records(
        'statistic', 
        cached=True, 
        filter=(('Category', 'WBT'),)
    )
    wbts, _ = map_concurrent(
        load_competition_best_times, 
        dict(zip(wbt_stats.description, zip(wbt_stats.url))),
        show_progress=False, 
    )
    wbts = pd.concat(wbts, names=['CompetitionType'])\
        .reset_index(0)\
        .reset_index(drop=True)
    wbts.CompetitionType = wbts.CompetitionType.str.extract(
        r": ([a-zA-Z]+)"
    )
    wbts.ResultTime = pd.to_timedelta(wbts.ResultTime)
    return wbts


def get_world_best_times():
    return get_competition_best_times().\
        sort_values('ResultTime').\
        groupby('BoatClass').\
        first()


def find_world_best_time(
        boat_class=None, 
        event=None, 
        race=None, 
        race_id=None, 
):
    if boat_class is None:
        if event is None:
            if race is None:
                if race_id is None:
                    raise ValueError(
                        'must pass at least one of '
                        'boat_class, event, race or race_id'
                        )
                race = get_worldrowing_data(
                    'race', race_id
                )
            event = get_worldrowing_data(
                'event', race['eventId']
            )
        boat_class = get_worldrowing_data(
            'boatClass', event['boatClassId']
        )['DisplayName']

    return get_world_best_times().loc[boat_class]