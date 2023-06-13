
from datetime import datetime, timedelta, date
import shutil
import urllib
import json
from pathlib import Path
import re
import argparse
from functools import lru_cache, partial
import logging
import sys

import requests
import pandas as pd
import numpy as np
import getpass

from . import geodesy, utils, files, splits

logger = logging.getLogger(__name__)

session_fields = [
    'crews.crew_users.user.crew_user_groups',
    'crews.crew_users.user:fields(id|first_name|last_name|full_name|picture)',
    'crews.gmt_score',
    'crews.inventory.usage_history',
    'participants.user.crew_user_groups',
    'participants.user.user_squads',
    'participants.user:fields(id|first_name|last_name|full_name|picture)',
    'session.structure.pieces.crews.crew_users.user.crew_user_groups',
    'session.structure.pieces.crews.crew_users.user:fields(id|first_name|last_name|full_name|picture)',
    'session.structure.pieces.crews.gmt_score',
    'session.structure.pieces.crews.inventory.usage_history',
    'session.structure.pieces.crews:order(created_at|desc)',
    'session_files.crews.session_crew_files.user',
    'session_files.users:fields(id|full_name)',
    'user_actions'
]


@lru_cache
def get_script():
    r = requests.get("https://app.ludum.com/")
    r.raise_for_status()
    script_endpt = re.search(
        rb"<script src=(\/js\/app\.[^>]+)>", r.content
    ).group(1).decode()
    headers = {
        "Referer": "https://app.ludum.com/login",
        "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="101", "Google Chrome";v="101"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36"
    }
    r = requests.get(
        "https://app.ludum.com" + script_endpt,
        headers=headers,
    )
    r.raise_for_status()
    return r.text


def get_client_secret():
    load_script = get_script()
    return re.search(r'client_secret:"([^"]+)"', load_script).group(1)


def get_client_id():
    load_script = get_script()
    return re.search(r'client_id:"([^"]+)"', load_script).group(1)


class LudumClient(utils.CachedClient):
    timeout = 10
    https_base_url = "https://api.ludum.com"
    _default_headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://app.ludum.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/101.0.4951.67 Safari/537.36"
        ),
    }
    concurrent_kws = {
        "max_workers": 4,
    }

    def __init__(
            self, username=None, password=None, club_id=None, client_secret=None,
            local_cache=None, path="ludum_data", map_kws=None,
    ):
        self.client_secret = client_secret
        self.club_id = club_id
        self.n_requests = 0
        self.session = requests.session()
        self._login_response = None

        super().__init__(
            username=username,
            password=password,
            path=path,
            local_cache=local_cache,
            map_kws=map_kws
        )

    @classmethod
    def from_credentials(cls, credentials, **kwargs):
        if not isinstance(credentials, dict):
            with open(credentials, 'r') as f:
                credentials = json.load(f)

        return cls(**credentials, **kwargs)

    def login(self, username=None, password=None, client_secret=None, club_id=None):
        username = username or self.username
        password = password or self.password
        client_secret = client_secret or self.client_secret
        club_id = club_id or self.club_id

        if username is None:
            print("please input your Ludum username/email: ")
            username = input()

        if password is None:
            password = getpass.getpass('Input your Ludum password: ')

        if client_secret is None:
            client_secret = get_client_secret()

        if club_id is None:
            self.club_id = get_client_id()

        self.authenticate(username, password, client_secret, self.club_id)

        return self

    def authenticate(self, username, password, client_secret, club_id):
        payload = {
            "username": username,
            "password": password,
            "client_secret": client_secret,
            "client_id": club_id,
            "grant_type": "password"
        }
        s = self.session.post(
            self.https_base_url + "/api/v2/login",
            data=payload,
            headers=self._default_headers,
            timeout=self.timeout
        )
        s.raise_for_status()

        self._login_response = s.json()

    @property
    def access_token(self):
        if self._login_response is None:
            self.login()

        return self._login_response['data']['accessToken']['access_token']

    def prepare_headers(self, **kwargs):
        headers = self._default_headers.copy()
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        if self.club_id:
            headers["club_id"] = self.club_id
            headers["club-id"] = self.club_id

        headers.update(kwargs)
        return headers

    def post(self, endpt, data=None, json=None, **kwargs):
        kwargs['headers'] = self.prepare_headers(**kwargs.get('headers', {}))
        self.n_requests += 1
        return self.session.post(
            self.https_base_url + endpt,
            data=data,
            json=json,
            timeout=self.timeout,
            **kwargs
        )

    def post_json(self, endpt, **kwargs):
        r = self.post(endpt, **kwargs)
        r.raise_for_status()
        return r.json()

    def get(self, endpt, **kwargs):
        kwargs['headers'] = self.prepare_headers(**kwargs.get('headers', {}))
        self.n_requests += 1
        return self.session.get(
            self.https_base_url + endpt,
            timeout=self.timeout,
            **kwargs
        )

    def get_json(self, endpt, **kwargs):
        r = self.get(endpt, **kwargs)
        r.raise_for_status()
        return r.json()

    def cached_get(self, key, endpt, local_cache=None, path=None, reload=False, **kwargs):
        local_cache = local_cache or self.local_cache
        path = path or self.path
        if local_cache:
            if reload:
                return local_cache.update(
                    key, path, self.get_json, endpt, **kwargs
                )
            else:
                return local_cache.get(
                    key, path, self.get_json, endpt, **kwargs
                )

        return self.get_json(endpt, **kwargs)

    def cached_post(self, key, endpt, local_cache=None, path=None, reload=False, **kwargs):
        local_cache = local_cache or self.local_cache
        path = path or self.path
        if local_cache:
            if reload:
                return local_cache.update(
                    key, path, self.post_json, endpt, **kwargs
                )
            else:
                return local_cache.get(
                    key, path, self.post_json, endpt, **kwargs
                )

        return self.post_json(endpt, **kwargs)

    def download_json(self, url, **kwargs):
        with requests.get(url, **kwargs) as r:
            self.n_requests += 1
            r.raise_for_status()
            try:
                return r.json()
            except requests.exceptions.JSONDecodeError:
                return {}

    def cached_json(self, key, url, local_cache=None, path=None, reload=False, **kwargs):
        local_cache = local_cache or self.local_cache
        path = path or self.path

        if self.local_cache:
            if reload:
                return local_cache.update(
                    key, path, self.download_json, url, **kwargs
                )
            else:
                return self.local_cache.get(
                    key, path, self.download_json, url, **kwargs
                )

        return self.download_json(url, **kwargs)

    def get_session_data(self, session_id, reload=True, **kwargs):
        fields = ",".join(session_fields)
        endpt = (
            f"/api/v2/session/{session_id}"
            "?fields=id|name|seat_racing|session"
            f"&include={fields}"
        )
        return self.cached_get(
            session_id, endpt, **kwargs, path=self.path / "session", reload=True)

    def get_date_agenda(self, date, reload=True, **kwargs):
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        payload = {
            "club_id": self.club_id,
            "start_date": date_str,
            "end_date": date_str,
            "display": "all-sessions",
        }
        payload.update(kwargs)
        return self.cached_post(
            date_str,
            "/api/v2/agenda?fields=id|session_id|start_date|end_date|sport|name|all_day_event|attendance_limit|location",
            data=payload,
            path=self.path / "agenda",
            reload=reload,
        )

    def get_agenda(self, start=None, end=None, period="60d", **kwargs):
        if not end:
            end = datetime.today()
        if not start:
            start = pd.to_datetime(end) - pd.to_timedelta(period)

        start, end = pd.to_datetime(start), pd.to_datetime(end)
        dates = pd.date_range(start, end)
        logger.info(
            "getting agenda from %s to %s", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        agenda, errors = utils.map_concurrent(
            self.get_date_agenda,
            dict(zip(dates, zip(dates))),
            **self.concurrent_kws
        )
        errors and logger.warning("get_agenda had errors %r", errors)
        return agenda

    def _get_agenda(self, start=None, end=None, days=60, **kwargs):
        if end:
            end_datetime = pd.to_datetime(end)
        else:
            end_datetime = datetime.today()

        if start:
            start_datetime = pd.to_datetime(start)
            if not end:
                end_datetime = start_datetime + timedelta(days=days)
        else:
            start_datetime = end_datetime - timedelta(days=days)

        payload = {
            "club_id": self.club_id,
            "start_date": start_datetime.strftime("%Y-%m-%d"),
            "end_date": end_datetime.strftime("%Y-%m-%d"),
            "display": "all-sessions",
        }
        payload.update(kwargs)
        r = self.post(
            "/api/v2/agenda?fields=id|session_id|start_date|end_date|sport|name|all_day_event|attendance_limit|location",
            data=payload
        )
        r.raise_for_status()
        return r.json()

    def load_agenda(self, start=None, end=None, days=60, **kwargs):
        data = self.get_agenda(
            start=start, end=end, days=days, **kwargs
        )
        agenda_data = sum(
            (d['data']['data'] for d in data.values()), []
        )
        return pd.json_normalize(agenda_data)

    def load_sessions(self, start=None, end=None, days=60, reload=True, **kwargs):
        agenda = self.load_agenda(
            start=start, end=end, days=days, **kwargs
        )
        logger.info("loading %d sessions", len(agenda))
        sessions, errors = self.map_concurrent(
            self.get_session_data,
            dict(zip(agenda.id, zip(agenda.id))),
            reload=True,
            **self.concurrent_kws
        )
        errors and logger.warning("load_sessions experienced %r", errors)
        return agenda, sessions

    def load_session_activity_files(
            self, session_id, user_id,
            gps_file=None, gps_file_url=None, hr_file=None, hr_file_url=None,
            **kwargs
    ):
        urls = {
            "gps_data": gps_file,
            "gps_processed": gps_file_url,
            "hr_data": hr_file,
            "hr_processed": hr_file_url,
        }
        loaded = (
            (k, self.cached_json(("activity", session_id, user_id, k), url))
            for k, url in urls.items() if url
        )
        return {
            k: activity for k, activity in loaded if activity
        }

    def load_session_activities_files(self, sessions):
        arg_names = (
            'session_id', 'user_id', 'gps_file', 'gps_file_url', 'hr_file', 'hr_file_url'
        )
        inputs = (
            (k, [data.get(c) for c in arg_names]) for k, data in sessions.items()
        )
        inputs = {k: args for k, args in inputs if any(args[2:])}
        logger.info("loading %d activities", len(inputs))
        activity_files, errors = self.map_concurrent(
            self.load_session_activity_files, inputs,
            **self.concurrent_kws,
        )
        errors and logger.warning(
            "load_session_activities_files experienced %r", errors)
        return activity_files

    def load_session_gps_data(self, session_id, user_id, gps_file):
        gps_data = self.load_session_activity_files(
            session_id, user_id, gps_file)['gps_data']
        return parse_ludum_gps_data(gps_data)

    def load_sessions_gps_data(self, sessions):
        arg_names = (
            'session_id', 'user_id', 'gps_file'
        )
        inputs = (
            (k, [data.get(c) for c in arg_names]) for k, data in sessions.items()
        )
        inputs = {k: args for k, args in inputs if args[2]}
        logger.info("loading %d gps tracks", len(inputs))
        activities, errors = self.map_concurrent(
            self.load_session_gps_data, inputs,
            **self.concurrent_kws,
        )
        errors and logger.warning(
            "load_sessions_gps_data experienced %r", errors)
        metadata = pd.concat({
            k: data[0] for k, data in activities.items()
        }, axis=0).droplevel(1, axis=0)
        positions = pd.concat({
            k: data[1] for k, data in activities.items()
        }, axis=0).dropna(subset=['latitude', 'longitude'])
        return metadata, positions

    def load_morning_monitoring(
        self, dates, ids=(), squad_ids=(), **kwargs
    ):
        load_date = partial(
            self.get_morning_monitoring, ids=ids, squad_ids=squad_ids
        )
        mm_data, errors = utils.map_concurrent(
            load_date,
            dict(zip(dates, zip(dates))),
            path=self.path / "morning_monitoring",
            **self.concurrent_kws,
            **kwargs
        )
        errors and logger.warning(
            "load_morning_monitoring dataframe errors %r", errors)
        return pd.concat(mm_data, axis=0)

    def get_morning_monitoring(self, date, ids=(), squad_ids=(), path=None):
        json_data = self.cached_post(
            date,
            "/api/v2/morning-monitoring",
            json={
                "date": date,
                "club_id": self.club_id,
                "athlete_tag_ids": list(ids),
                "squad_tag_ids": list(squad_ids),
                "include": "user:fields(id|first_name|last_name|full_name)"
            },
            headers={
                "Content-Type": "application/json",
            },
            path=path
        )
        data = pd.json_normalize([
            {
                **{
                    k: d[k] for k in [
                        'athlete_id', 'username', 'athlete_name'
                    ]
                },
                **d['daily'][date]
            }
            for d in json_data['data']
        ])
        data['date'] = date
        return data

    def get_squads(self, **kwargs):
        params = {
            'club_id': self.club_id,
            'include': (
                'squad_users.user:fields(id%7Cfirst_name%7Clast_name%7Cfull_name),'
                'squad_users.user.current_club_user'
            ),
            'limit': '1000'
        }
        params.update(kwargs)
        data = self.cached_get(
            "squad", "/api/v2/squads", params=params,
        )
        squads = {}
        for squad_data in data['data']['data']:
            name = squad_data['name']
            squad_data['squad_users'] = pd.json_normalize(
                squad_data['squad_users']['data'])
            squads[name] = squad_data

        return squads

    def get_club_users(self, **kwargs):
        params = {
            'fields': 'first_name|last_name|full_name|id',
            'club_id': self.club_id,
            'include': 'current_club_roles,current_club_user',
            'limit': '1000'
        }
        params.update(kwargs)
        user_data = self.cached_get(
            "club_users", "/api/v2/club-users", params=params)
        return pd.json_normalize(user_data['data']['data'])

    def get_ergo_data(self, start_date=None, end_date=None, period="365d", distances=(2000, 5000), times=(), **kwargs):
        end_date = pd.to_datetime(end_date or pd.Timestamp.today())
        start_date = pd.to_datetime(
            start_date or end_date - pd.to_timedelta(period))
        params = {
            "club_id": self.club_id,
            "start_date": "2017-01-01",
            "end_date": end_date.strftime("%Y-%m-%d"),
            "distances": distances,
            "times": times,
            **kwargs
        }
        headers = {'Content-Type': 'application/json'}
        raw_data = self.cached_post(
            "ergo_data", "/api/v2/ergo-report", json=params, headers=headers
        )
        ergo_data = pd.concat({
            athlete['user_name']: pd.json_normalize(sum(
                (event['pieces'] for event in athlete['events'].values()),
                []
            ))
            for athlete in raw_data['data']
        }, axis=0).droplevel(1)
        ergo_data.index.name = "Name"
        return ergo_data


def extract_session_data(sessions):
    session_info = extract_session_info(sessions)

    session_files = (
        (session, session_data['data']['session_files'])
        for session, session_data in sessions.items()
        if 'session_files' in session_data['data']
    )
    session_files = pd.concat({
        session: pd.DataFrame.from_records([
            {
                **{'full_name': d['full_name']},
                **dict(utils.flatten_json(d['session_individual_files']['data']))
            }
            for d in data.get('data', {}).get('users', {}).get('data', []) if d
        ])
        for session, data in session_files
    }, axis=0, ignore_index=False)

    for c in session_files.columns[
        session_files.columns.str.endswith('id') &
        ~ session_files.columns.str.endswith('guid')
    ]:
        session_files[c] = session_files[c].fillna(0).astype(int)

    session_files = session_files.copy()

    session_files['session_name'] = session_info.name.loc[session_files.session_id].values
    return session_info.copy(), session_files.copy()


def extract_session_info(sessions):
    return pd.concat(
        [pd.json_normalize(data['data']) for i, data in sessions.items()],
        ignore_index=True
    ).set_index('id')


def extract_session_files(sessions):
    session_files = (
        (session, session_data['data']['session_files'])
        for session, session_data in sessions.items()
        if 'session_files' in session_data['data']
    )
    session_files = pd.concat({
        session: pd.DataFrame.from_records([
            {
                **{'full_name': d['full_name']},
                **dict(utils.flatten_json(d['session_individual_files']['data']))
            }
            for d in data.get('data', {}).get('users', {}).get('data', []) if d
        ])
        for session, data in session_files
    },
        axis=0, ignore_index=False, names=["activity_id", ""]
    ).droplevel(1)

    for c in session_files.columns[
        session_files.columns.str.endswith('id') &
        ~ session_files.columns.str.endswith('guid')
    ]:
        session_files[c] = session_files[c].fillna(0).astype(int)

    return session_files.copy()


def extract_crew_member(session_info):
    crew_info = pd.concat(
        dict(session_info.loc[
            session_info['crews.data'].apply(len) > 0, 'crews.data'
        ].apply(pd.json_normalize).items()),
        names=["session_id", ""]
    ).droplevel(1).reset_index().set_index("id")

    crew_info["boat_class"] = crew_info["weight.data.name"].replace({
        "Heavyweight": "",
        "Lightweight": "L"
    }).str.cat(
        [crew_info["gender.data.short"], crew_info["boat_type.data.name"]]
    )
    crew_info['gmt'] = pd.to_timedelta(
        crew_info["gmt_score.data.gmt_score"], unit='s')

    member_info = pd.concat(
        dict(crew_info.loc[
            crew_info['crew_users.data'].apply(len) > 0, 'crew_users.data'
        ].apply(pd.json_normalize).items()),
        ignore_index=True
    ).set_index('id')
    return crew_info, member_info


def extract_rowers(sessions):
    return pd.concat(
        {
            (rower['user_id'], session_id):
            pd.json_normalize({**crew, **rower})
            for session_id, data in sessions.items()
            for crew in data['data']['crews']['data']
            for rower in crew['crew_users']['data']
        },
        names=['user_id', 'session_id']
    ).droplevel(-1)


def extract_activity_info(agenda, sessions):
    session_info = extract_session_info(sessions)
    session_files = extract_session_files(sessions)
    crew_info, member_info = extract_crew_member(session_info)
    return merge_activity_info(agenda, session_info, session_files, crew_info, member_info)


def merge_activity_info(agenda, session_info, session_files, crew_info, member_info):
    activity_info = session_files.join(
        member_info.join(
            crew_info[
                ["session_id", "name", "boat_class", "gmt", "parent_id"]
            ].rename(columns={"name": "boat_name"}),
            on='crew_id'
        ).set_index(["session_id", "user_id"])[
            ["boat_name", "boat_class", "gmt"]
        ],
        on=['activity_id', "user_id"]
    ).join(
        agenda.set_index("id")[[
            "start_date", "name"
        ]],
        on="activity_id"
    )
    return activity_info


def merge_index(index, activity_info):
    index_frame = index.to_frame().reset_index(drop=True)
    merged_index = index_frame.join(
        activity_info[[
            "id", "name", "start_date", "full_name", "boat_name", "boat_class", "gmt"
        ]].set_index("id"),
        on='activity_id'
    ).rename(columns={
        "name": "session",
        "boat_name": "boat",
        "full_name": "athlete",
    })
    return pd.MultiIndex.from_frame(merged_index[
        ["start_date", 'session', "boat_class", "gmt", "boat", 'athlete']
        + list(c for c in index_frame.columns if c != "activity_id")
    ])


def find_best_times(positions, activity_info, cols=None, max_distance=20, max_group=4, max_order=20):
    session_positions = positions.groupby(level=0)
    logger.info(
        "finding best times for %d sessions with %d points",
        session_positions.ngroups, len(positions)
    )
    session_best_times, errors = utils.map_concurrent(
        splits.find_all_best_times,
        session_positions,
        singleton=True,
        total=session_positions.ngroups,
        cols=cols,
        threaded=False,
    )

    best_times = pd.concat(
        session_best_times,
        names=['activity_id', 'length', 'distance']
    ).reset_index()
    best_times['order'] = best_times.groupby(
        ["activity_id", "length"]).cumcount() + 1
    best_times = best_times.set_index([
        "activity_id", "length", "order"
    ]).unstack(level=[1, 2]).T.unstack(level=0)

    gmts = activity_info.loc[
        activity_info.id.isin(best_times.columns.levels[0])
    ].set_index(
        "id"
    ).gmt.dropna()
    best_splits = best_times.xs("split", level=1, axis=1)[gmts.index].apply(
        lambda s: s.fillna(pd.Timedelta(0)).dt.total_seconds()
    )
    pgmts = (
        gmts.dt.total_seconds().values[None] / best_splits / 4
    ).replace(np.inf, 0)

    best_times = pd.concat([
        best_times, pd.concat({"pgmt": pgmts}, axis=1).swaplevel(axis=1)
    ], axis=1)

    best_times.columns.names = 'activity_id', 'data'

    best_times.columns = best_times.columns.to_frame().reset_index(drop=True).join(
        activity_info[[
            "id", "name", "start_date", "full_name", "boat_name", "boat_class", "gmt"
        ]].set_index("id"),
        on='activity_id'
    ).rename(columns={
        "name": "session",
        "boat_name": "boat",
        "full_name": "athlete",
    }).set_index([
        "start_date", 'session', "boat_class", "gmt", "boat", 'athlete', 'data'
    ]).index

    distance_orders = (
        (s, d, i * max_group + j) for i in range(max_order // max_group)
        for s, d in splits._STANDARD_DISTANCES.items() for j in range(max_group)
    )
    distance_orders = pd.MultiIndex.from_tuples([
        (s, n + 1) for s, d, n in distance_orders
        if n * d <= max_distance
    ], names=["interval", "rank"])
    return best_times.sort_index(
        axis=1, ascending=[False] + [True] * (best_times.columns.nlevels - 1)
    ).reindex(distance_orders)


def find_crossings(positions, activity_info, loc=None):
    locations = splits.load_location_landmarks(loc)
    session_positions = positions.groupby(level=0)
    logger.info(
        "finding crossings for %d sessions with %d points",
        session_positions.ngroups, len(positions)
    )
    session_crossing_data, errors = utils.map_concurrent(
        splits.find_all_crossing_data,
        session_positions,
        singleton=True,
        total=session_positions.ngroups,
        cols=["cadence", "bearing", "heart_rate"],
        locations=locations,
        threaded=False,
    )
    errors and logger.warning("find_crossings experienced errors %r", errors)

    crossing_data = pd.concat(
        session_crossing_data,
        names=['activity_id', 'leg', 'location', "landmark", 'distance']
    ).reset_index("distance")
    crossing_data.columns.name = "data"

    crossing_data['leg_time'] = (
        crossing_data.time
        - crossing_data.groupby(level=[0, 1]
                                ).time.min().reindex(crossing_data.index)
    )

    zero_index = crossing_data.leg_time == pd.Timedelta(0)

    crossing_data['timeDelta'] = (
        crossing_data.time - crossing_data.time.shift()
    )
    crossing_data.loc[zero_index, "timeDelta"] = pd.Timedelta(0)

    crossing_data['distanceDelta'] = crossing_data.distance.diff()
    crossing_data.loc[zero_index, "distanceDelta"] = 0
    crossing_data['split'] = pd.to_timedelta(
        crossing_data.timeDelta.dt.total_seconds() / crossing_data.distanceDelta / 2,
        unit='s'
    )
    crossing_data.loc[zero_index, "split"] = pd.Timedelta(0)

    crossings = crossing_data[[
        'time', 'leg_time', 'distanceDelta', 'timeDelta', 'split',
        'cadence', 'bearing', "heart_rate"
    ]].stack("data").unstack(locations.index.names).reindex(
        locations.index, axis=1
    ).T
    crossings.columns = merge_index(crossings.columns, activity_info)
    return crossings.swaplevel(-2, -1, axis=1).sort_index(
        axis=1, ascending=[False] + [True] * (crossings.columns.nlevels - 1)
    )


def load_morning_monitoring(
    api: LudumClient, start, end=None, squad_name=None, squad_id=None, **kwargs
):
    if squad_name:
        squads = api.get_squads()
        squad_id = squads[squad_name]['id']
    if squad_id:
        kwargs['squad_ids'] = [squad_id]

    if end is None:
        end = date.today()

    dates = [
        date.strftime("%Y-%m-%d") for date in pd.date_range(start=start, end=end)
    ]

    mm_cols = [
        'date', 'athlete_id', 'username', 'athlete_name',
        'standing_hr', 'sleep_quality', 'sleep_time', 'percived_shape',
        'weight', 'temperature', 'waking_hr', 'healthy', 'ill', 'injured',
    ]
    logger.info("loading morning_monitoring from %s to %s",
                dates[0], dates[-1])
    mm_data = api.load_morning_monitoring(dates, **kwargs)
    return mm_data.loc[
        mm_data.athlete_entered_monitoring, mm_cols
    ].set_index(["athlete_name", "date"]).sort_index()


def get_parser():
    parser = argparse.ArgumentParser(
        description='Analyse recent ludum data'
    )
    utils.add_credentials_arguments(parser)
    parser.add_argument(
        '--actions',
        choices=['morning_monitoring', "best_times", "crossings", "reload"],
        default=['morning_monitoring'],
        nargs='+',
        help='specify action will happen'
    )
    parser.add_argument(
        '--excel-file',
        type=Path, default='ludum.xlsx', nargs='?',
        help='path of output excel spreadsheet'
    )
    utils.add_gspread_arguments(parser)
    parser.add_argument(
        '--folder',
        type=str, default=None, nargs='?',
        help='folder path to download fit files'
    )
    parser.add_argument(
        '--start-date',
        type=pd.to_datetime,
        help='start date to search for activities from in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--end-date',
        type=pd.to_datetime,
        default=date.today(),
        help='start date to search for activities from in YYYY-MM-DD format'
    )
    parser.add_argument(
        "--period",
        type=pd.to_timedelta,
        default="1w",
        help="Period to search back from"
    )
    parser.add_argument(
        "--login",
        type=bool,
        default=True,
        help="Whether to log in to Ludum or just use the local cache"
    )
    utils.add_logging_argument(parser)
    return parser


def run(args):
    options = get_parser().parse_args(args)
    utils.set_logging(options)

    local_cache = utils.json_cache if options.folder else None

    if options.credentials:
        api = LudumClient.from_credentials(
            options.credentials,
            local_cache=local_cache,
            path=options.folder
        )
    else:
        api = LudumClient(
            username=options.user,
            password=options.password,
            local_cache=local_cache,
            path=options.folder
        )

    if options.login:
        api.login()
    else:
        logger.info("using local cached data")

    if not options.start_date:
        options.start_date = options.end_date - options.period

    morning_monitoring = "morning_monitoring" in options.actions
    calc_best_times = "best_times" in options.actions
    calc_crossings = "crossings" in options.actions
    reload = "reload" in options.actions

    agenda = None
    sessions = None
    activity_info = None
    positions = None
    metadata = None
    athlete_data = None
    best_times = None
    crossings = None

    if morning_monitoring:
        athlete_data = load_morning_monitoring(
            api, options.start_date, options.end_date,
        )

    if calc_best_times or calc_crossings or reload:
        agenda, sessions = api.load_sessions(
            options.start_date, options.end_date, reload=options.login
        )

    if calc_best_times or calc_crossings:
        activity_info = extract_activity_info(agenda, sessions)
        metadata, positions = api.load_sessions_gps_data(
            activity_info.loc[activity_info.sport_name ==
                              "Rowing"].set_index("id").T
        )

    if calc_best_times:
        try:
            best_times = find_best_times(
                positions, activity_info, cols=[
                    'cadence', 'bearing', 'heart_rate']
            )
        except Exception as e:
            logger.warning("find_best_times experienced error %r", e)
            best_times = e

    if calc_crossings:
        try:
            crossings = find_crossings(positions, activity_info)
        except Exception as e:
            logger.warning("find_crossings experienced error %r", e)
            best_times = e

    if any([morning_monitoring, calc_best_times, calc_crossings]):
        if options.excel_file.name:
            logger.info("saving results to %s", options.excel_file)
            with pd.ExcelWriter(options.excel_file) as xlf:
                if isinstance(athlete_data, pd.Dataframe):
                    athlete_data.to_excel(xlf, "morning_monitoring")

                if isinstance(best_times, pd.Dataframe):
                    best_times.to_excel(xlf, "best_times")

                if isinstance(crossings, pd.Dataframe):
                    crossings.to_excel(xlf, "crossings")

        if options.gspread:
            spread = options.gspread
            logger.info("saving results to %s", spread)
            if isinstance(athlete_data, pd.DataFrame):
                utils.to_gspread(athlete_data, spread, "morning_monitoring")

            if isinstance(best_times, pd.DataFrame):
                gsheet_best_times = utils.format_gsheet(best_times)
                gsheet_best_times.loc[
                    :, (slice(None),) * 6 + ("pgmt",)
                ] = gsheet_best_times.loc[
                    :, (slice(None),) * 6 + ("pgmt",)
                ].fillna(0).applymap("{:.2%}".format).replace("0.00%", "")
                utils.to_gspread(gsheet_best_times.T, spread, "best_times")

            if isinstance(crossings, pd.DataFrame):
                gsheet_crossings = utils.format_gsheet(
                    crossings.swaplevel(-2, -1,
                                        axis=1).sort_index(axis=1, ascending=False)
                )
                for loc in gsheet_crossings.index.levels[0]:
                    loc_crossings = gsheet_crossings.loc[loc].T
                    if not loc_crossings.empty:
                        utils.to_gspread(loc_crossings, spread, loc)

    return dict(
        api=api,
        options=options,
        agenda=agenda,
        sessions=sessions,
        activity_info=activity_info,
        positions=positions,
        metadata=metadata,
        athlete_data=athlete_data,
        best_times=best_times,
        crossings=crossings,
    )


def main():
    try:
        run(sys.argv[1:])
    except Exception as err:
        logging.error(err)

    input("Press enter to finish")


if __name__ == "__main__":
    main()


def download_fit(url):
    return files.parse_fit_data(requests.get(url, stream=True).raw.read())


def download_ludum_data(row, path='ludum_data', overwrite=False, file_cols=None):
    data_folder = Path(path)
    if file_cols is None:
        file_cols = [
            'session_individual_files_data_gps_file',
            'session_individual_files_data_gps_file_url',
            'session_individual_files_data_hr_file',
            'session_individual_files_data_hr_file_url'
        ]
    paths = []
    for c in file_cols:
        url = row[c]
        out_path = data_folder / urllib.parse.urlparse(row[c]).path.strip("/")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        paths.append(out_path)
        if overwrite or not out_path.exists():
            with requests.get(url, stream=True) as r, open(out_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

    return paths


def download_all_ludum_data(session_files, path='ludum_data', overwrite=False):
    return utils.map_concurrent(
        download_ludum_data,
        {
            (row.session_id, row.full_name): (row,)
            for _, row in session_files.iterrows()
        },
        max_workers=4,
        overwrite=False
    )


def parse_ludum_gps_data(data):
    data = data.copy()

    data.pop("line")
    positions = pd.DataFrame.from_dict(
        data.pop('position'), orient='index'
    ).reset_index(drop=True)
    last = positions.index[-1]
    positions['time'] = pd.to_datetime(positions.timestamp, unit='s')

    if 'lat' in positions:
        positions['latitude'] = positions.lat
        positions['longitude'] = positions.get("lng", positions.get("long"))
        positions = positions.dropna(
            subset=['time', 'latitude', 'longitude', 'distance']
        ).reset_index(drop=True)
        positions['distance'] /= 1000
        positions['distanceDelta'] = - positions.distance.diff(-1)
        positions.loc[last, 'distanceDelta'] = 0
        positions['bearing_r'] = geodesy.rad_bearing(
            positions, positions.shift(-1))
        positions.loc[last, 'bearing_r'] = positions.bearing_r.iloc[-2]
        positions['bearing'] = np.rad2deg(positions.bearing_r)

    positions['timeElapsed'] = positions.time - positions.time[0]
    positions['timeDelta'] = - positions.timeElapsed.diff(-1)
    positions.loc[last, 'timeDelta'] = pd.Timedelta(0)
    positions.index.name = 'datapoint'

    metadata = pd.json_normalize(data)
    return metadata, positions


def read_ludum_data(filepath):
    with open(filepath, "r") as f:
        return parse_ludum_gps_data(json.load(f))


def read_ludum_path(path, allowed=None, max_workers=10, **kwargs):
    id_paths = (
        ((int(p.parent.parent.stem), int(p.parent.stem), ), p)
        for p in Path(path).glob("**/processed_data.json")
    )
    if allowed is not None:
        activity_paths = {
            index: (p,) for index, p in id_paths if index in allowed
        }
    else:
        activity_paths = {index: (p,) for index, p in id_paths}

    activity_data, read_ludum_data_errors = utils.map_concurrent(
        read_ludum_data, activity_paths, max_workers=max_workers, **kwargs
    )
    activity_positions = pd.concat({
        index: data for index, (_, data) in activity_data.items() if not data.empty
    }, names=['data_session_id', 'user_id', 'data_point'])
    activity_info = pd.concat({
        index: info for index, (info, _) in activity_data.items() if not info.empty
    }, names=['data_session_id', 'user_id']).droplevel(-1)
    return activity_positions, activity_info, read_ludum_data_errors
