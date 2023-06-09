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
BOATCLASSES = {
    '02316e75-fdf2-4660-af40-48dda1867e1f': 'M2-',
    '079e71bb-98cd-47f4-8ca4-e3fd0cdbc538': 'JM1x',
    '07c32034-57cf-46cd-a4c5-2fe8dbb0aa25': 'LTAID4+',
    '0f218eeb-6f65-48c7-941c-97f0901cb8cd': 'BLW2-',
    '1163bebd-2c43-441f-97d5-7f2bd57fd062': 'M2+',
    '1238bd03-cde0-4760-a34b-6e745ffb0814': 'M4+',
    '15e1ef74-79c6-4227-96f1-86d793efbf5b': 'PR3 M2-',
    '16946db9-72f2-4375-a797-a26fd485dd26': 'W2x',
    '1797cb82-0ced-4fb3-9acd-4a6646f824cb': 'JM4+',
    '182d0aae-7e73-4900-ae8e-f1aafbe5c48a': 'LM1x',
    '21b5a4c6-1ee1-4867-9d5b-c5b62f0e0c5e': 'ASW1x',
    '25baa6f6-7830-4d2f-9b1f-f0662a283f01': 'CJW2x',
    '26af8860-5146-422d-bb10-3eef8b28d883': 'JW2x',
    '27d5614e-6530-49d1-acf5-da132b71a45d': 'M4-',
    '28dd4be4-2458-4251-ba49-d17f75c12b66': 'TAMix1x',
    '2a2a5aa7-8592-4684-99ce-4f02679061e6': 'BM8+',
    '2a5dc948-808f-4391-bc38-c1b24566f1ed': 'LTAMix4x+',
    '2b2bfa01-a902-4ad8-be87-4527c9ba7e6d': 'W4-',
    '2b6d4e9c-3a41-4654-8bc1-45bfd289af94': 'CM2x',
    '2d21e07e-0fc5-4b1a-ae10-d811487e45d6': 'CJMix2x',
    '2fc7bcd7-3ccc-49eb-ab18-50ab09ae501d': 'BM4-',
    '3304a580-b3b1-4b4e-b51f-e4f63d2e5853': 'BW2-',
    '33c0a216-0ab8-4838-8f7c-6b15c2b13e53': 'CMix2x',
    '3830f652-0963-4d75-a081-a4ef4553a10c': 'BM2-',
    '3aba91c2-a8a8-4fe4-8148-efd036ef7c3a': 'LTAMix4+',
    '3f0d0a7d-92a6-4c53-90f5-7c8f01972964': 'M1x',
    '410492b4-4ae6-4c71-998b-7c69325c510b': 'CJM2x',
    '43b5679c-8f9e-45f9-8111-fca22bdc02cc': 'LW1x',
    '4458faa3-d55b-495f-8018-51250f82e5ba': 'LM2x',
    '46564a8c-b4b3-4bb3-bbf4-cd96bb4ee4f5': 'LM2-',
    '4b0b81cf-08fd-4cdd-80f5-44848b9517e4': 'CM1x',
    '4bbcf35e-b424-4000-9f4f-048f2edce263': 'JW2-',
    '4f77e783-c93e-4692-9e52-0ece558a898e': 'LM4-',
    '5083723f-8e0b-4b92-88fb-1ab05b074d59': 'W4x+',
    '532b264c-47e1-4d35-928a-85d9733928d0': 'M8+',
    '5851b9f9-5240-4fea-a19e-43b24eb99a10': 'JM2-',
    '5911da97-fe4a-42ce-8431-7f3aa9cdfb20': 'JW4+',
    '59a7dd8d-3374-40bb-898e-cc9919407bf2': 'LW4-',
    '5a308214-114a-4b49-b985-fdc6e0b7b319': 'BW2x',
    '5e834702-14c3-4a85-b38d-9afb419f5294': 'W2-',
    '5e9e5c6d-ef18-4475-b91e-e0fa899a47a2': 'CM4x+',
    '60cecbef-919c-47c0-9f16-0a8edd105d82': 'BLW1x',
    '61be8c32-b56d-4cac-bf56-46246fb1c349': 'LW2x',
    '65d140a8-79dd-4475-b16e-b16a39fe054e': 'LTAMix2x',
    '683b227a-51fd-4c09-a6a0-460ca3711b08': 'JW1x',
    '698a8cfe-1ca4-4b42-9801-92adab701075': 'TAMix2x',
    '69baef35-d07e-4a2d-bbdc-3c2eebcef471': 'CJW1x',
    '6b7c0445-c065-49cd-90b3-4c56317e4346': 'BM1x',
    '6edc9d42-edda-4f32-9469-91d54e0eaeb1': 'JW4x',
    '7249e207-5c76-4c58-9c7c-b7e2a4c74b50': 'W4x',
    '731e3206-45c5-4596-b48e-5a21b1675918': 'PR2 M1x',
    '74864b35-be40-431a-a48c-f1b605a6759e': 'BM2x',
    '78b65b8c-c20e-46f8-99ac-f654a0ea17b2': 'JM2+',
    '78f004f6-cadd-4d0f-804b-21cf26458355': 'JM2x',
    '7b8c2fd0-0286-4bd3-a5c3-0f17a4bc3bd0': 'LW2-',
    '81fdafb0-750a-4f6f-b382-c00a82def40c': 'BM2+',
    '843310e2-f28c-41a7-90e9-b6f8c289b154': 'PR2 Mix2x',
    '8557de5c-a78a-4917-9e6d-4be118dc075b': 'W4+',
    '8661c474-9506-473a-9784-d00ecfa2167f': 'JM4-',
    '871abbbb-2413-45b5-b83c-03619c6f0ee5': 'BW8+',
    '8a078108-6741-4ad1-ab5a-704194538227': 'W1x',
    '8d8bb18e-f1b8-4048-ace1-65c27d03cab1': 'CW2x',
    '8df04fba-76af-40c4-ac38-73f377bfb258': 'BLM4x',
    '94013194-9368-4d4d-a9b0-c99e9fd8feb3': 'CW1x',
    '98f584fc-4987-499c-b831-a3550e1a5498': 'ASM1x',
    '9abcfecb-bf04-4773-bd30-3a45357259d4': 'BLM2-',
    '9cb5d841-32a6-4388-8a30-6d7f51bdfdbc': 'M2x',
    'a3fffabb-8d13-499c-b381-b74da92fd320': 'BLM1x',
    'a7d04c48-a5a0-42d5-9343-c68395337ddf': 'BLM4-',
    'a93ad830-0374-4950-9cf7-d0c6b1db1363': 'IDMix4+',
    'aee5f04d-6939-4173-a829-999dfb77f159': 'BLW2x',
    'b16ded37-785e-4638-adf2-7f3165a8d7b6': 'LM8+',
    'b61f430f-d4ee-48e0-a632-95b9194ccb16': 'PR1 W1x',
    'b6bcf280-a57c-4a6a-96fb-91e6a1355002': 'PR3 Mix4+',
    'b834436f-4378-4825-aaa0-f9905959abdb': 'W8+',
    'ba397032-a5c6-466a-ba1b-23e521237048': 'LM4x',
    'ba9814b9-0243-445c-979d-189da9ad8407': 'BW4x',
    'be5a6d89-d55b-4316-9366-7ed1ef0f307b': 'BM4x',
    'bed56642-164b-4ea5-93ce-1fcadd0ef017': 'JM8+',
    'bf36bd84-191f-4dd6-8837-6b0151662598': 'JW4x+',
    'c18403e0-53cf-4b6b-a1f1-60ad2b220c5a': 'BLW4x',
    'c47c6b93-dd0d-4645-9545-2ee853db284c': 'LW4x',
    'c5810370-41ac-45c2-aeff-01dcb1d6d078': 'BW4-',
    'cc8022e2-3ee1-463a-bddf-aa115d68e704': 'BLM2x',
    'd4140064-67dc-47d0-957a-a42e90bf8433': 'M4x',
    'd7365b40-b544-480b-aa99-b0ffad7a13e9': 'PR1 M1x',
    'd97e6295-e55a-4235-abc3-e986dead5137': 'JW8+',
    'dc9439cd-36d9-47fd-a205-bc2612f4f83b': 'JM4x',
    'dd434535-b9eb-4b38-9ecf-f3fd6dc9cadf': 'PR2 W1x',
    'e22e0fe4-66bc-42f1-82f4-162bb1cc384b': 'BW1x',
    'e23cc24f-4a62-4de3-ab99-50a40c8416f0': 'PR3 Mix2x',
    'e784fbb3-b45f-45d5-be67-0699cdba4553': 'BM4+',
    'e972ad8f-b73e-41ef-a30b-dee4a391a734': 'BW4+',
    'e9c10080-48fa-4a7f-aec8-1f90ecdc6744': 'PR3 W2-',
    'ea76db60-d6cb-4dd8-a35a-23b2af3042fe': 'CW4x+',
    'eaf690e9-cc36-45bc-b59f-6d510c16daf7': 'CMix4x+',
    'f1123970-de2b-462e-89a5-f74e0b447939': 'LW8+',
    'f7576033-2298-4549-ad10-cdbfc78417b0': 'CJM1x',
    'ff2aa19c-db81-4c8e-a523-b7a3ef50ce31': 'JW4-'
}

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

def _rsc_phase_names():
    phase_name = {
        'FNL-': "Final", 
        'HEAT': "Heat", 
        'REP-': "Repechage", 
        'SFNL': "Semifinal", 
        'PREL': "Preliminary",
        'QFNL': "Quarterfinal"
    }
    default_codes = {f"000{i + 1}00": "" for i in range(26)}
    final_codes = {f"000{i + 1}00": " " + chr(i + 65) for i in range(26)}
    semi_codes = {
        f"000{2 * i + 1}00": " " + chr(2 * i + 65) + "/" + chr(2 * i + 66) 
        for i in range(13)
    }
    semi_codes.update(
        (f"000{2 * i + 2}00", " " + chr(2 * i + 65) + "/" + chr(2 * i + 66) )
        for i in range(13)
    )
    phase_codes = {
        'FNL-': final_codes, 
        'SFNL': semi_codes, 
    }
    return {
        p + c: name + code
        for p, name in phase_name.items()
        for c, code in phase_codes.get(p, default_codes).items()
    }

PHASE_NAMES = _rsc_phase_names()

def parse_race_codes(race_codes):
    race_codes = pd.Series(race_codes)
    codes = race_codes.str.extract(
        r"ROW([MWX])([A-Z]+[0-9])-+([A-Z0-9]*)-+([A-Z]+-?[0-9]+)", 
        expand=True
    )
    return pd.concat({
        "Gender": codes[0].replace({
            "W": "Women",
            "M": "Men",
            "X": "Mixed",
        }),
        "Category": codes[2].replace({
            "": "Open", 
            "L": "Lightweight", 
        }),
        "Phase": codes[3].replace(PHASE_NAMES)
    }, axis=1)


def rename_column(s, prefix=''):
    c = f"{prefix}.{s}"
    if c.endswith("DisplayName"):
        c = c[:-12]

    return c


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

def request_worldrowing(*endpoints, request_kws=None, **kwargs):
    return requests.get(*prepare_request(*endpoints, **kwargs), **dict(request_kws or {}))

def request_worldrowing_json(*endpoints, request_kws=None, **kwargs):
    return load_json_url(*prepare_request(*endpoints, **kwargs), **dict(request_kws or {}))


cached_request_worldrowing_json = cache(request_worldrowing_json)


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
        data = cached_request_worldrowing_json(*endpoints, **hashable_kws)
    else:
        data = request_worldrowing_json(*endpoints, **kwargs, request_kws=request_kws)

    return data.get("data", [])


def get_worldrowing_record(*endpoints, **kwargs):
    data = get_worldrowing_data(*endpoints, **kwargs)
    return pd.json_normalize(data).loc[0].rename(data.get("id", data.get("DisplayName", "")))


def get_worldrowing_records(*endpoints, request_kws=None, **kwargs):
    records = get_worldrowing_data(*endpoints, **kwargs, request_kws=request_kws)
    if not records:
        return pd.DataFrame([])
    
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

def parse_race(race):
    return parse_races(race.to_frame().T).iloc[0]

def parse_races(races):
    race_codes = parse_race_codes(races['RscCode'])
    boat_classes = races[
        'event.boatClassId'
    ].replace(BOATCLASSES).rename("boatClass.DisplayName")
    races['Day'] = races.Date.dt.date
    return pd.concat([races, race_codes, boat_classes], axis=1) 


def get_competition_races(competition_id=None, cached=True):
    competition_id = competition_id or get_most_recent_competition().name
    races = get_worldrowing_records(
            "race",
            cached=cached,
            filter=(("event.competitionId", competition_id),),
            sort=(("eventId", "asc"), ("Date", "asc")),
            include=(
                "event.competition,raceStatus,racePhase,"
                "raceBoats.raceBoatIntermediates.distance,"
                "event"
            )
        )
    return parse_races(races) 

def get_race(race_id, **kwargs):
    kwargs.setdefault(
        "sort", (("eventId", "asc"), ("Date", "asc"))
    )
    kwargs.setdefault(
        "include", 
        "event.competition,raceStatus,racePhase,"
        "raceBoats.raceBoatIntermediates.distance,event"
    )
    race = get_worldrowing_record(
        "race", race_id, **kwargs
    )
    return parse_races(race.to_frame().T).iloc[0]

def get_competitions(year=None, fisa=True, has_results=True, cached=True, **kwargs):
    
    kwargs['filter'] = tuple(dict(kwargs.get('filter', ())).items())
    if year is not False:
        kwargs['filter'] += ('Year', year or datetime.date.today().year),
    if fisa:
        kwargs['filter'] += ('IsFisa', 1),
    if has_results:
        kwargs['filter'] += ('HasResults', (1 if has_results else '')),
    
    kwargs.setdefault("sort", {"StartDate": "asc"})
    kwargs['include'] = ",".join(set(
        kwargs.get('include', "CompetitionType").split(",")
        + "competitionType,venue".split(",")
    ))
    competitions = get_worldrowing_records("competition", cached=cached, **kwargs)
    competitions['CompetitionType'] = competitions['competitionType.DisplayName'].replace(
        COMPETITION_TYPES
    )

    return competitions


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
        race_data.distance = race_data.distance.str.extract("([0-9]+)")[0].astype(float)
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

    race_data = race_data.dropna(subset=['distance'])
    race_data.distance = race_data.distance.astype(int)

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
