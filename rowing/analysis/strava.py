
import streamlit as st

import time

import pandas as pd

import stravalib

from rowing.app import inputs
from rowing.analysis import files


@st.cache_resource
def get_client(code):
    client = stravalib.client.Client()
    client.code = code
    return client


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
        client = stravalib.client.Client()
        authorize_url = client.authorization_url(
            client_id=st.secrets.strava['client_id'],
            redirect_uri=inputs.get_url(),
            scope=['read_all', 'profile:read_all', 'activity:read_all']
        )
        st.page_link(
            authorize_url,
            label='Authorise Strava Access',
            icon=":material/link:"
        )


@st.cache_data
def get_activities(code, end=None, start=None, limit=None):
    client = get_client(code)
    activities = pd.json_normalize([
        a.to_dict() for a in client.get_activities(end, start, limit)
    ])
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
    })
    activity_data['latitude'] = activity_data.latlng.str[0]
    activity_data['longitude'] = activity_data.latlng.str[1]
    activity_data[
        'time'
    ] = pd.to_timedelta(activity_data.time, unit='s') + activity.start_date
    return files.process_latlontime(activity_data).drop(
        columns=['latlng']
    )
