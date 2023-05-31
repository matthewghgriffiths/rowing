import datetime
from collections.abc import Mapping
import logging

import pandas as pd

from .utils import (
    getnesteditem,
    extract_fields,
    read_times,
    map_concurrent,
    CURRENT_TIMEZONE,
    format_timedelta_hours,
    format_totalseconds,
    merge,
    cache,
    _pyodide
)

if _pyodide:
    use_requests = False
    import pyodide
    from js import XMLHttpRequest
    import json
    from urllib import parse
else:
    use_requests = True
    import requests

logger = logging.getLogger("world_rowing.api")

OLYMPIC_BOATCLASS = [
    "M1x",
    "LM2x",
    "W1x",
    "M2x",
    "M2-",
    "LW2x",
    "M4-",
    "W1x",
    "W2x",
    "W2-",
    "M4x",
    "M8+",
    "W4x",
    "W8+",
    "W4-",
]

RACE_PHASES = {
    "Repechage": "0959f5e8-f85a-40fb-93ab-b6c477f6aade",
    "Test Race": "92b34c4e-af58-4e91-8f4a-22c09984a006",
    "Heat": "cd3d5ca1-5aed-4146-b39b-a192ae6533f1",
    "Final": "e0fc3320-cd66-43af-a5b5-97afd55b2971",
    "Semifinal": "e6693585-d2cf-464c-9f8e-b2e531b26400",
}

COMPETITION_TYPES = {
    'Elite Overall': 'Elite Overall',
    'Olympic Games': 'Olympic Games',
    'Paralympics': 'Paralympic Games',
    'World Rowing Championships': 'World Rowing Championships',
    'World Rowing Cup I': 'World Rowing Cup',
    'World Rowing Cup II': 'World Rowing Cup',
    'World Rowing Cup III': 'World Rowing Cup',
    'World Rowing  U23 Championships': 'World Rowing U23 Championships',
    'World Rowing Under 19 Championships': 'World Rowing Under 19 Championships',
    'European Rowing Championships': 'European Rowing Championships',
    'Adaptive World Rowing Championships': 'World Rowing Championships',
    'World Rowing Championships  International Competitions': 'World Rowing Championships  International Competitions',
}

WBT_TYPES = {
    'World best time for: EliteECH': 'European Rowing Championships',
    'World best time for: EliteOverall': 'Elite Overall',
    'World best time for: EliteWC': 'World Rowing Cup',
    'World best time for: EliteWCH': 'World Rowing Championships  International Competitions',
    'World best time for: EuropeanRowingChampionships': 'European Rowing Championships',
    'World best time for: OlympicGames': 'Olympic Games',
    'World best time for: ParaOverall': 'World Rowing Championships',
    'World best time for: ParalympicGames': 'Paralympic Games',
    'World best time for: Under19': 'World Rowing Under 19 Championships',
    'World best time for: Under23': 'World Rowing U23 Championships',
    'World best time for: WorldRowingCup': 'World Rowing Cup'
}

def stringify_value(value):
    if isinstance(value, (str, int, float)):
        return str(value)
    else:
        return "||".join(map(str, value))


def prepare_options(key, options):
    if isinstance(options, str):
        return ((key, options),)

    _options = options.items() if isinstance(options, Mapping) else options
    return ((f"{key}[{field}]", stringify_value(value)) for field, value in _options)


def prepare_params(**kwargs):
    return {
        field: value
        for key, options in kwargs.items()
        for field, value in prepare_options(key, options)
    }


if use_requests:
    def load_json_url(url, params=None, timeout=20., **kwargs):
        logger.debug("requesting: %s\nparams: %s", url, params)
        r = requests.get(url, params=params, timeout=timeout, **kwargs)
        r.raise_for_status()
        if r.text:
            return r.json()
        else:
            return {}


else:

    def load_json_url(url, params=None, **kwargs):
        if params:
            query = parse.urlencode(params)
            url = parse.urlparse(url)._replace(query=query).geturl()

        # Some requests only work asyncronously
        req = XMLHttpRequest.new()
        req.open("GET", url, False)
        req.send(None)
        if req.response:
            return json.loads(req.response)
        else:
            return {}


def prepare_request(*endpoints, **kwargs):
    endpoint = "/".join(endpoints)
    url = f"https://world-rowing-api.soticcloud.net/stats/api/{endpoint}"
    params = prepare_params(**kwargs) or None
    logger.debug("preparing: %s\nparams: %s", url, params)
    return url, params


def request_worldrowing_data(*endpoints, request_kws=None, **kwargs):
    return load_json_url(*prepare_request(*endpoints, **kwargs), **dict(request_kws or {}))


cached_request_worldrowing_data = cache(request_worldrowing_data)


@cache
def load_competition_best_times(json_url, **kwargs):
    return pd.DataFrame.from_records(
        extract_fields(record, WBT_RECORDS)
        for record in load_json_url(json_url, **kwargs)["BestTimes"]
    )


def get_worldrowing_data(*endpoints, cached=True, request_kws=None, **kwargs):
    if cached:
        hashable_kws = {
            k: vals
            if isinstance(vals, (str, int, float, tuple))
            else tuple(dict(vals).items())
            for k, vals in kwargs.items()
        }
        data = cached_request_worldrowing_data(*endpoints, **hashable_kws)
    else:
        data = request_worldrowing_data(*endpoints, **kwargs, request_kws=request_kws)

    return data.get("data", [])


def get_worldrowing_record(*endpoints, **kwargs):
    data = get_worldrowing_data(*endpoints, **kwargs)
    return pd.Series(data, name=data.pop("id", data.get("DisplayName", "")))


def get_worldrowing_records(*endpoints, request_kws=None, **kwargs):
    records = get_worldrowing_data(*endpoints, **kwargs, request_kws=request_kws)
    records = pd.concat(
        [pd.json_normalize(r) for r in records]
    )
    if "id" in records.columns:
        records.set_index("id", inplace=True)

    for col in records.columns:
        if "Date" in col:
            dates = pd.to_datetime(records[col])
            if dates.notna().all():
                records[col] = dates

    return records


def get_competition_events(competition_id=None, cached=True):
    competition_id = competition_id or get_most_recent_competition().name
    return get_worldrowing_records(
        "event",
        cached=True,
        filter=(("competitionId", competition_id),),
        # sort=(("Date", "asc"),),
    )


def get_competition_races(competition_id=None, cached=True):
    competition_id = competition_id or get_most_recent_competition().name
    races = get_worldrowing_records(
            "race",
            cached=cached,
            filter=(("event.competitionId", competition_id),),
            sort=(("eventId", "asc"), ("Date", "asc")),
            include=(
                "event.competition,raceStatus,racePhase,"
                "raceBoats.raceBoatIntermediates.distance"
            )
        )

    return races 


def get_competitions(year=None, fisa=True, has_results=True, cached=True, **kwargs):
    year = year or datetime.date.today().year
    kwargs['filter'] = tuple(dict(kwargs.get('filter', ())).items())
    kwargs['filter'] += ('Year', year),
    kwargs['filter'] += ('IsFisa', (1 if fisa else "")),
    kwargs['filter'] += ('HasResults', (1 if has_results else '')),
    kwargs.setdefault("sort", {"StartDate": "asc"})
    kwargs['include'] = ",".join(set(
        kwargs.get('include', "CompetitionType").split(",")
        + "competitionType,venue".split(",")
    ))
    events = get_worldrowing_records("competition", cached=cached, **kwargs)

    return events


def get_this_years_competitions(fisa=True, has_results=True):
    year = datetime.date.today().year
    return get_competitions(year=year, fisa=fisa, has_results=has_results)


def get_last_years_competitions(fisa=True, has_results=True):
    year = datetime.date.today().year
    return get_competitions(year=year - 1, fisa=fisa, has_results=has_results)


def get_most_recent_competition(fisa=True):
    competitions = get_this_years_competitions(fisa=fisa)
    started = competitions.StartDate < datetime.datetime.now()
    if not started.any():
        competitions = get_last_years_competitions(fisa=fisa)
        started = competitions.StartDate < datetime.datetime.now()

    competition = competitions.loc[started].iloc[-1]
    logger.info(f"loaded most recent competition: {competition.DisplayName}")
    return competition


def get_live_races(fisa=True, competition=None):
    if competition is None:
        competition = get_most_recent_competition(fisa)

    return get_worldrowing_records(
        "race",
        cached=False,
        filter=(
            ("event.competitionId", competition.name),
            ('raceStatus.displayName', 'LIVE'),
        ),
        include='event.competition,raceStatus',
        sort=(("eventId", "asc"), ("Date", "asc")),
    )

def get_live_race(fisa=True, competition=None):
    finished = False 
    live_races = get_live_races(fisa=fisa, competition=competition)
    for race_id, race in live_races.iterrows():
        data = get_worldrowing_data(
            'livetracker', "live", race_id, cached=False
        )
        lanes = data['config'].get("lanes", [])
        started = (
            data.get('live') 
            or any(lane.get("live") for lane in lanes)
        )
        
        if started:
            live_race = race 
            
            finished = all(lane['_finished'] for lane in lanes)
            if not finished:
                break 
        elif finished:
            break
        else:
            pass
            
    else:
        return None
    return live_race


def get_last_race_started(fisa=True, competition=None):
    races = get_last_races(n=1, fisa=fisa, competition=competition)
    if races is not None:    
        race = races.iloc[0]
        logger.info(f"loaded last race started: {race.DisplayName}")
        return race

get_most_recent_race = get_last_race_started

def get_last_races(n=1, fisa=True, competition=None):
    if competition is None:
        competition = get_most_recent_competition(fisa)
    races = get_competition_races(competition.name)
    if not races.empty:
        started = races.DateString < datetime.datetime.now().astimezone()
        return races.loc[started].sort_values("DateString").iloc[-n:]


def get_next_races(n=1, fisa=True, competition=None):
    if competition is None:
        competition = get_most_recent_competition(fisa)
    races = get_competition_races(competition.name)
    to_race = races.DateString > datetime.datetime.now().astimezone()
    return races.loc[to_race].sort_values("DateString").iloc[:n]


def show_next_races(n=10, fisa=True, competition=None):
    next_races = get_next_races(n, fisa=fisa, competition=competition)[
        ["DisplayName", "DateString"]
    ].reset_index(drop=True)
    next_races.DateString = next_races.DateString.dt.tz_convert(
        CURRENT_TIMEZONE)
    next_races.columns = ["Race", "Time"]
    now = datetime.datetime.now().astimezone(CURRENT_TIMEZONE)
    next_races["Time to race"] = (
        next_races.Time - now).apply(format_timedelta_hours)
    return next_races


def get_boat_types():
    return get_worldrowing_records("boatClass", cached=True)


def get_competition_types():
    return get_worldrowing_records("competitionType", cached=True)


def get_statistics():
    return get_worldrowing_records("statistic", cached=True)


def get_venues():
    return get_worldrowing_records("venue", cached=True)


RACEBOAT_FIELDS = {
    "id": ("id",),
    "raceId": ("raceId",),
    "boatId": ("boatId",),
    "Country": ("DisplayName",),
    "Rank": ("Rank",),
    "Lane": ("Lane",),
    "ResultTime": ("ResultTime",),
}


def get_race_results(
    race_id=None, event_id=None, competition_id=None, cached=True, **kwargs
):
    filters = tuple(dict(kwargs.pop("filter", ())).items())
    if race_id:
        filters += (("id", race_id),)
    if event_id:
        filters += (("eventId", event_id),)
    if competition_id:
        filters += (("event.competitionId", competition_id),)

    race_data = get_worldrowing_data(
        "race",
        cached=cached,
        filter=filters,
        include="raceBoats.raceBoatIntermediates",
        **kwargs,
    )
    results = pd.DataFrame.from_records(
        [
            extract_fields(boat, RACEBOAT_FIELDS)
            for race in race_data
            for boat in race["raceBoats"]
            # if boat['ResultTime']
        ]
    )
    if len(results):
        results.set_index(["raceId", "id"], inplace=True)
        if "ResultTime" in results.columns:
            results.ResultTime = read_times(results.ResultTime)
    return results


INTERMEDIATE_FIELDS = {
    "id": ("id",),
    # 'raceId': 'raceId',
    "raceBoatId": ("raceBoatId",),
    "Rank": ("Rank",),
    "ResultTime": ("ResultTime",),
    "distanceId": ("distanceId",),
    "distance": ("distance", "DisplayName"),
}


def get_intermediate_results(
    race_id=None, event_id=None, competition_id=None, cached=True, **kwargs
):
    filters = tuple(dict(kwargs.pop("filter", ())).items())
    if race_id:
        filters += (("id", race_id),)
    if event_id:
        filters += (("eventId", event_id),)
    if competition_id:
        filters += (("event.competitionId", competition_id),)

    race_data = get_worldrowing_data(
        "race",
        cached=cached,
        filter=filters,
        include="raceBoats.raceBoatIntermediates.distance",
        **kwargs,
    )
    # race_fields = RACEBOAT_FIELDS.copy()
    # race_fields['raceBoatId'] = race_fields.pop('id')
    results = pd.DataFrame.from_records(
        [
            {
                **extract_fields(boat, RACEBOAT_FIELDS),
                **extract_fields(inter, INTERMEDIATE_FIELDS),
            }
            for race in race_data
            for boat in race["raceBoats"] if boat["ResultTime"]
            for inter in sorted(
                boat["raceBoatIntermediates"], key=lambda x: x["ResultTime"]
            )
        ]
    )

    if len(results):
        results.set_index(["raceId", "raceBoatId", "id"], inplace=True)
        results.ResultTime = pd.to_timedelta(results.ResultTime)
    return results


WBT_RECORDS = {
    "BoatClass": ("BoatClass",),
    "ResultTime": ("Competitor", "ResultTime"),
    "Competition": ("Competition", "Name"),
    "Venue": ("Venue", "Name"),
    "Event": ("Event", "DisplayName"),
    "EventId": ("Event", "Id"),
    "Race": ("Race", "Name"),
    "RaceId": ("Race", "Id"),
    "Country": ("Competitor", "Nationality", "Name"),
    "Date": ("DateOfBT",),
}


def _extract_wbt_record(record):
    return {key: getnesteditem(record, *items) for key, items in WBT_RECORDS.items()}


@cache
def get_competition_best_times(timeout=2., load_event_info=False):
    wbt_stats = get_worldrowing_records(
        "statistic", cached=True, filter=(("Category", "WBT"),)
    )
    wbts, _ = map_concurrent(
        load_competition_best_times,
        dict(zip(wbt_stats.description, zip(wbt_stats.url))),
        progress_bar=None,
        timeout=timeout, 
    )
    wbts = (
        pd.concat(wbts, names=["CompetitionType"]).reset_index(
            0).reset_index(drop=True)
    )
    
    wbts['CompetitionType.name'] = wbts.CompetitionType
    wbts['CompetitionType.DisplayName'] = wbts.CompetitionType.str.extract(r": ([a-zA-Z]+)")
    wbts['CompetitionType'] = wbts.CompetitionType.replace(WBT_TYPES)
    wbts.ResultTime = pd.to_timedelta(wbts.ResultTime)
    wbts['time'] = wbts.ResultTime.dt.total_seconds().apply(format_totalseconds)


    if load_event_info:
        event_information = pd.concat(map_concurrent(
            lambda event_id: pd.json_normalize(get_worldrowing_data(
                "event", event_id, include="competition.competitionType"
            )),
            {i: (i,) for i in wbts.EventId}
        )[0]).reset_index(1, drop=True)
        return wbts.join(event_information, on='EventId')
    
    return wbts


def get_world_best_times():
    return (
        get_competition_best_times()
        .sort_values("ResultTime")
        .groupby("BoatClass")
        .first()
    )


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
                        "must pass at least one of "
                        "boat_class, event, race or race_id"
                    )
                race = get_worldrowing_data("race", race_id)
            event = get_worldrowing_data("event", race["eventId"])
        boat_class = get_worldrowing_data("boatClass", event["boatClassId"])[
            "DisplayName"
        ]

    return get_world_best_times().loc[boat_class]

def get_raw_competition_results(competition_id=None, with_intermediates=True):
    competition_id = competition_id or get_most_recent_competition().name
    competition_races = get_competition_races(competition_id).reset_index().add_prefix("race.")
    competition_events = get_competition_events(competition_id).add_prefix("event.")

    if with_intermediates:
        race_results = get_intermediate_results(competition_id=competition_id)
    else:
        race_results = get_race_results(competition_id=competition_id)

    boat_classes = get_boat_types().reset_index().add_prefix("boatClass.")
    wbts = get_world_best_times().ResultTime
    wbts.index.name, wbts.name = "id", "worldBestTime"

    return merge_competition_results(
        race_results,
        competition_races, 
        competition_events, 
        boat_classes, 
        wbts, 
    )

def merge_competition_results(
        race_results,
        competition_races, 
        competition_events, 
        boat_classes, 
        wbts, 
):
    GMT_col = wbts.name
    boat_wbts = pd.merge(
        boat_classes.reset_index(), wbts, 
        left_on='boatClass.DisplayName', right_on=wbts.index.name
    )
    event_wbts = pd.merge(
        competition_events.reset_index(), boat_wbts, 
        left_on='event.boatClassId', right_on='boatClass.id', 
        # suffixes = ('_event', '_boat'),
    )
    race_wbts = pd.merge(
        competition_races, event_wbts,
        left_on='race.eventId', right_on='id', 
        # suffixes = ('_race', '_event')
    )
    race_data = pd.merge(
        race_results.loc[
            race_results.Rank.notna()
            & (race_results.ResultTime > pd.Timedelta(0))
        ], 
        race_wbts,
        left_on='raceId',  right_on='race.id' 
    )
    race_data.Rank = race_data.Rank.astype(int)
    if "distance" in race_data.columns:
        race_data.distance = race_data.distance.str.extract("([0-9]+)")[0].astype(int)
        race_data['PGMT'] = (
            race_data[GMT_col].dt.total_seconds() * race_data.distance 
            / race_data.ResultTime.dt.total_seconds() / 2000
        )
    else:
        race_data['PGMT'] = race_data[GMT_col].dt.total_seconds() / race_data.ResultTime.dt.total_seconds()
    race_data[GMT_col] = race_data[GMT_col].dt.total_seconds().apply(format_totalseconds)
    race_data['time'] = race_data.ResultTime
    race_data.ResultTime = race_data.ResultTime.dt.total_seconds().apply(format_totalseconds)
    
    for col in race_data.columns[race_data.columns.str.contains("DisplayName")]:
        n = col.rsplit(".")[-2]
        race_data[n] = race_data[col]
    
    return race_data

def get_competition_results(competition_id=None, with_intermediates=True):
    race_data = get_raw_competition_results(
        competition_id=competition_id,
        with_intermediates=with_intermediates
    )
    if with_intermediates:
        race_summaries = race_data.set_index(
            ['DisplayName_boat', 'DisplayName_event', 'DisplayName', 'Progression', 'distance', 'Rank']
        )[['Country', 'Lane', 'ResultTime', 'PGMT']].sort_index()
        race_summaries.index.names = [
            'BoatClass', 'Event', 'Race', 'Progression', 'Distance', 'Rank'
        ]
    else:
        race_summaries = race_data.set_index(
            ['DisplayName_boat', 'DisplayName_event', 'DisplayName', 'Progression', 'Rank']
        )[['Country', 'Lane', 'ResultTime', 'worldBestTime', 'PGMT', 'Date']].sort_index()
        race_summaries.index.names = [
            'BoatClass', 'Event', 'Race', 'Progression', 'Rank'
        ]
    return race_summaries

def get_competition_pgmts(competition_id=None, finals_only=False):
    competition_id = competition_id or get_most_recent_competition().name
    competition_races = get_competition_races(competition_id)
    competition_events = get_competition_events(competition_id)
    competition_results = get_race_results(competition_id=competition_id)
    boat_classes = get_boat_types()
    wbts = get_world_best_times().ResultTime
    wbts.index.name, wbts.name = "id", "worldBestTime"

    if competition_results.empty:
        return competition_results

    if finals_only:
        competition_races = competition_races.loc[
            competition_races.racePhaseId == RACE_PHASES["Final"]
        ]

    competition_results = merge(
        (
            competition_results.reset_index(),
            competition_races[["eventId", "Date"]],
            competition_events["boatClassId"],
            boat_classes,
            wbts,
        ),
        left_on=("raceId", "eventId", "boatClassId", "DisplayName"),
        right_on="id",
    )
    competition_results["PGMT"] = (
        competition_results.worldBestTime / competition_results.ResultTime
    )

    results = competition_results.loc[
        competition_results.ResultTime > pd.to_timedelta(0),
        [
            "DisplayName",
            "PGMT",
            "ResultTime",
            "worldBestTime",
            "Country",
            "Rank",
            "Lane",
            "Date",
        ],
    ]
    results.columns = ["Boat", "PGMT", "Time",
                       "WBT", "Country", "Rank", "Lane", "Date"]
    # results['PGMT'] = results.worldBestTime / results.Time
    results["Rank"] = results.Rank.astype(int)
    results.Time = results.Time.dt.total_seconds().apply(format_totalseconds)
    results.WBT = results.WBT.dt.total_seconds().apply(format_totalseconds)
    return results.sort_values("PGMT", ascending=False).reset_index(drop=True)
