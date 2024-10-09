
import streamlit as st
import os
import io
from pathlib import Path
import base64
import time

import pandas as pd

from rowing.app import inputs
from rowing.analysis import files


_file_path = Path(os.path.abspath(__file__))
_module_path = _file_path.parent
_DATA_PATH = _module_path.parent.parent / 'data'

_STRAVA_SVG = _DATA_PATH / "btn_strava_connectwith_orange.svg"


@st.cache_resource
def get_client(code):
    import stravalib
    client = stravalib.client.Client()
    client.code = code
    return client


def _strave_base64_logo(svg_path=_STRAVA_SVG):
    with open(svg_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


_URL_SVG_HTML = """<a target="{}" href="{}">
<img src="data:image/svg+xml;base64,{}" height="48px">
</a>"""


def _strava_html_logo(url, svg_path=_STRAVA_SVG, target='_self'):
    return _URL_SVG_HTML.format(target, url, _strave_base64_logo(svg_path))


def connect_client():
    if 'code' in st.query_params:
        code = st.query_params['code']
        client = get_client(code)
        try:
            if client.access_token is None:
                print("Exchanging code for token")
                response = client.exchange_code_for_token(
                    client_id=st.secrets.strava['client_id'],
                    client_secret=st.secrets.strava['secret'],
                    code=code
                )
                client.access_token = response['access_token']
                client.refresh_token = response['refresh_token']
                client.expires_at = response['expires_at']

            if time.time() > int(client.expires_at or 0):
                print('Token has expired, will refresh')
                response = client.refresh_access_token(
                    client_id=st.secrets.strava['client_id'],
                    client_secret=st.secrets.strava['secret'],
                    refresh_token=client.refresh_token
                )
                client.access_token = response['access_token']
                client.refresh_token = response['refresh_token']
                client.expires_at = response['expires_at']

            athlete = client.get_athlete()
            name = f"{athlete.firstname} {athlete.lastname}"
            print("Hello, {}".format(name))
        except Exception as e:
            print(e)
            st.query_params.pop("code")
            st.query_params['strava'] = 1
            st.rerun()

        return client
    else:
        try:
            import stravalib
            client = stravalib.client.Client()
            url = inputs.get_url()
            authorize_url = client.authorization_url(
                client_id=st.secrets.strava['client_id'],
                redirect_uri=url,
                scope=['read_all', 'profile:read_all', 'activity:read_all']
            )
            st.markdown(
                _strava_html_logo(
                    authorize_url, target='_blank' if 'streamlit' in url else '_self'),
                unsafe_allow_html=True,
            )
        except AttributeError:
            st.write("strava secrets not set yet")
        except ImportError:
            st.write("stravalib not installed yet")


@st.cache_data
def get_strava_activities(code, end=None, start=None, limit=None):
    client = get_client(code)
    return get_activities(client, end=end, start=start, limit=limit)


def get_activities(client, end=None, start=None, limit=None):
    if limit:
        start = end = None
    else:
        start, end = sorted(pd.to_datetime(
            [start, end]).to_pydatetime())
        limit = None

    activities = pd.json_normalize([
        a.dict(exclude=['map'])
        for a in client.get_activities(end, start, limit)
    ])
    for c in ['elapsed_time', 'moving_time']:
        if c in activities:
            activities[c] = pd.to_timedelta(activities[c], unit='s')
    if 'average_speed' in activities:
        activities['average_split'] = pd.to_timedelta(
            500 / activities['average_speed'], unit='s'
        )
    return activities


@st.cache_data
def load_strava_activity(code, activity_id):
    client = get_client(code)
    return load_activity(client, activity_id)


def load_activity(client, activity_id):
    activity = client.get_activity(activity_id)
    streams = client.get_activity_streams(activity_id)

    data = {
        k: list(v) for k, v in streams.items()
    }
    activity_data = pd.DataFrame({
        k: v[-1][1] for k, v in data.items()
    }).rename(columns={
        "heartrate": "heart_rate"
    })
    activity_data['latitude'] = activity_data.latlng.str[0]
    activity_data['longitude'] = activity_data.latlng.str[1]
    activity_data[
        'time'
    ] = (
        pd.to_timedelta(activity_data.time, unit='s')
        + pd.Timestamp(activity.start_date)
    )
    return files.process_latlontime(activity_data).drop(
        columns=['latlng']
    )


def strava_app():
    client = connect_client()
    if client is not None:

        athlete = client.get_athlete()

        name = f"{athlete.firstname} {athlete.lastname}"
        st.write(f"Hello {name}!")

        cols = st.columns(3)
        with cols[0]:
            limit = st.number_input(
                "How many activities to load, "
                "set to 0 if selecting date range",
                value=1,
                min_value=0,
                step=1
            )
        with cols[1]:
            date1 = st.date_input(
                "Select Date",
                value=pd.Timestamp.today() + pd.Timedelta("1d"),
                format='YYYY-MM-DD'
            )
        with cols[2]:
            date2 = st.date_input(
                "Range",
                value=pd.Timestamp.today() - pd.Timedelta("7d"),
                format='YYYY-MM-DD'
            )

        activities = get_strava_activities(client.code, date1, date2, limit)
        if activities.empty:
            return

        st.divider()
        activities['athlete'] = name
        activities['activity'] = activities['athlete'].str.cat(
            activities[['name', 'start_date_local']].astype(str),
            sep=' '
        )

        columns_order = [
            'activity',
            'sport_type',
            'start_date_local',
            'distance',
            'elapsed_time',
            'moving_time',
            'name',
            'average_split',
            # 'average_cadence',
            # 'average_heartrate',
            # 'description',
        ]
        activities['elapsed_time'] = (
            activities['elapsed_time'] + pd.Timestamp(0)).dt.time
        activities['moving_time'] = (
            activities['moving_time'] + pd.Timestamp(0)).dt.time
        activities['average_split'] = (
            activities['average_split'] + pd.Timestamp(0)).dt.time

        sel_activities = inputs.filter_dataframe(
            activities,
            key='filter strava activities',
            select_all=False,
            select_first=True,
            column_order=columns_order,
            column_config={
                "elapsed_time": st.column_config.TimeColumn(
                    format='h:mm:ss', disabled=True
                ),
                "moving_time": st.column_config.DatetimeColumn(
                    format='h:mm:ss', disabled=True
                ),
                "average_split": st.column_config.DatetimeColumn(
                    format='m:ss.S', disabled=True
                ),
            },
            disabled=activities.columns.difference(['select', 'activity']),
            modification_container=st.popover("Filter Activities"),
        )
        strava_data = {
            activity.activity: load_strava_activity(
                client.code, activity.id
            )
            for _, activity in sel_activities.iterrows()
        }
        if st.toggle("Download gpx data"):
            for activity, activity_data in strava_data.items():
                st.download_button(
                    f":inbox_tray: Download: {activity}.gpx",
                    io.BytesIO(
                        files.make_gpx_track(
                            activity_data).to_xml().encode()
                    ),
                    # type='primary',
                    file_name=f"{activity}.gpx",
                )

        return strava_data
