

import streamlit as st
import io
import functools

import pandas as pd

from rowing.app import inputs
from rowing.analysis import files, app
from rowing import utils

GARMIN_EPOCH = pd.Timestamp('1989-12-31 00:00:00')
UNIX_EPOCH = pd.Timestamp(0)
GARMIN_TIMESTAMP = (GARMIN_EPOCH - UNIX_EPOCH) // pd.Timedelta('1s')


def parse_garmin_fit_json(fit_json):
    positions = pd.DataFrame.from_records(
        fit_json).dropna(axis=1, how='all')
    positions['time'] = pd.to_datetime(
        positions.timestamp + GARMIN_TIMESTAMP, unit='s'
    )
    positions['distance'] /= 100
    if 'enhanced_speed' in positions.columns:
        positions['velocity_smooth'] = positions['enhanced_speed'] / 1000

    return files._parse_fit_positions(positions)


@st.cache_resource
def _client(username):
    import garminconnect
    client = garminconnect.Garmin(
        email=username
    )
    return client


def cache_client(func):
    @st.cache_data
    @functools.wraps(func)
    def cached_func(username, *args):
        client = _client(username)
        return func(client, *args)

    return cached_func


def prompt_mfa(container=None):
    def get_mfa():
        with container or st.container():
            return st.text_input("Enter MFA Code: ")

    return get_mfa


def login(user_container=None, pw_container=None, mfa_container=None):
    try:
        import garth
    except ImportError as e:
        print(e)
        return

    with user_container or st.container():
        username = st.text_input(
            "Enter email address: ", key='garmin_email')
    with pw_container or st.container():
        password = st.text_input("Enter password: ", type='password')

    client = _client(username)
    if client.garth.oauth1_token:
        print("already logged in")
        return client

    if username and password:
        try:
            client = _client(username)
            if client.garth.oauth1_token:
                print("already ")
                return client
            client.password = password
            client.prompt_mfa = prompt_mfa(mfa_container)
            if client.login():
                return client
            else:
                st.write(f"Could not log in {username}")
        except garth.exc.GarthHTTPError as e:
            print(e)


def get_activities(client, limit, *args):
    if limit:
        activities = client.get_activities(0, limit)
    elif len(args) == 2:
        start, end = pd.to_datetime(
            args).sort_values().strftime("%Y-%m-%d")
        activities = client.get_activities_by_date(start, end)
    elif len(args) == 1:
        date = pd.Timestamp(args[0]).strftime("%Y-%m-%d")
        activities = client.get_activities_fordate(date)
    else:
        activities = None

    if activities:
        activities = pd.json_normalize(activities)
        activities.startTimeLocal = pd.to_datetime(activities.startTimeLocal)
        activities['initials'] = activities.ownerFullName.map(utils.initials)
        activities['distance'] /= 1000
        activities.sort_values("startTimeLocal", ascending=True, inplace=True)
        activities['date'] = activities.startTimeLocal.dt.date
        activities['startTime'] = activities.startTimeLocal.dt.time
        activities['session'] = activities.groupby(['date']).cumcount() + 1
        activities.sort_values("startTimeLocal", ascending=False, inplace=True)

        activities['activity'] = activities.apply(
            "{0.initials} {0.date} #{0.session}".format,
            axis=1
        )
        activities['duration'] = pd.to_datetime(
            activities.duration, unit='s').dt.time
        activities['elapsedDuration'] = pd.to_datetime(
            activities.elapsedDuration, unit='s').dt.time
        activities['movingDuration'] = pd.to_datetime(
            activities.movingDuration, unit='s').dt.time
        activities['averageSplit'] = pd.to_datetime(
            500 / activities.averageSpeed.replace(0, float('nan')), unit='s').dt.time

        return activities


def download_fit(client, activity_id):
    zip_data = client.download_activity(
        activity_id, dl_fmt=client.ActivityDownloadFormat.ORIGINAL)
    return io.BytesIO(zip_data)


def load_fit(client, activity_id):
    data = files.zipfit_to_json(download_fit(client, activity_id))
    return parse_garmin_fit_json(data)


def download_gpx(client, activity_id):
    zip_data = client.download_activity(
        activity_id, dl_fmt=client.ActivityDownloadFormat.GPX)
    return io.BytesIO(zip_data)


get_garmin_activities = cache_client(get_activities)
load_garmin_fit = cache_client(load_fit)
download_garmin_gpx = cache_client(download_gpx)
download_garmin_fit = cache_client(download_fit)


def get_activity_hr(client, activity_id):
    hr = client.get_activity_hr_in_timezones(activity_id)
    hrz = pd.to_datetime(
        pd.json_normalize(hr).set_index(
            ['zoneNumber', 'zoneLowBoundary']
        ).secsInZone.rename(
            activity_id
        ), unit='s'
    ).dt.time
    return hrz.to_frame().T.rename_axis(
        'activityId'
    )


get_garmin_activity_hr = cache_client(get_activity_hr)


def get_activities_hr(client, activity_ids, max_workers=10):
    hrz, errors = utils.map_concurrent(
        get_activity_hr,
        {i: (client, i) for i in activity_ids},
        max_workers=max_workers
    )
    if errors:
        for k, e in errors:
            print(k)
            print(e)

    return pd.concat(hrz).droplevel(0)


def get_garmin_activities_hr(username, activity_ids, max_workers=10):
    username = getattr(username, "username", username)
    hrz, errors = utils.map_concurrent(
        get_garmin_activity_hr,
        {i: (username, i) for i in activity_ids},
        max_workers=max_workers
    )
    if errors:
        for k, e in errors:
            print(k)
            print(e)

    return pd.concat(hrz).droplevel(0)


SLEEP_COLS = [
    'restingHeartRate',
    'dailySleepDTO.sleepTimeSeconds',
    'dailySleepDTO.sleepStartTimestampLocal',
    'dailySleepDTO.sleepEndTimestampLocal',
    'dailySleepDTO.deepSleepSeconds',
    'dailySleepDTO.lightSleepSeconds',
    'dailySleepDTO.remSleepSeconds',
    'dailySleepDTO.awakeSleepSeconds',
    'dailySleepDTO.averageSpO2Value',
    'dailySleepDTO.lowestSpO2Value',
    'dailySleepDTO.highestSpO2Value',
    'dailySleepDTO.averageSpO2HRSleep',
    'dailySleepDTO.averageRespirationValue',
    'dailySleepDTO.lowestRespirationValue',
    'dailySleepDTO.highestRespirationValue',
    'dailySleepDTO.avgSleepStress',
]


def get_day_sleep_stats(client, day):
    day = pd.Timestamp(day)
    date = day.strftime("%Y-%m-%d")

    sleep = client.get_sleep_data(date)
    s = pd.json_normalize(sleep)
    s = s[s.columns.intersection(SLEEP_COLS)]
    s.columns = s.columns.str.removeprefix('dailySleepDTO.')
    return s


get_garmin_day_sleep_stats = cache_client(get_day_sleep_stats)

SLEEP_SECOND_COLS = [
    'sleepTimeSeconds', 'deepSleepSeconds',
    'lightSleepSeconds', 'remSleepSeconds',
    'awakeSleepSeconds'
]
SLEEP_SECOND_RENAME = {
    c: c.removesuffix("Seconds")
    for c in SLEEP_SECOND_COLS
}


def get_garmin_sleep_stats(username, start, end):
    start, end = pd.to_datetime([start, end]).sort_values()

    stats, errors = utils.map_concurrent(
        get_garmin_day_sleep_stats,
        {d: (username, d) for d in pd.date_range(start, end)},
    )
    if errors:
        print(errors)

    sleep_stats = pd.concat(stats, names=['day']).droplevel(1)

    for c in [
        'sleepStartTimestampLocal', 'sleepEndTimestampLocal'
    ]:
        sleep_stats[c] = pd.to_datetime(sleep_stats[c], unit='ms')

    for c in SLEEP_SECOND_COLS:
        sleep_stats[c] = pd.to_datetime(
            sleep_stats[c], unit='s').dt.time

    return sleep_stats.rename(columns=SLEEP_SECOND_RENAME)


ACTIVITY_FILTER_COLUMNS = [
    'activity',
    'date',
    'startTime',
    'session',
    'activityName',
    'distance',
    'duration',
    # 'elapsedDuration',
    'movingDuration',
    'averageSplit',
    'activityType.typeKey',
    'averageHR',
    'maxHR',
    'TimeInZone1',
    'TimeInZone2',
    'TimeInZone3',
    'TimeInZone4',
    'TimeInZone5',
    'ownerFullName',
]


def time_config():
    return st.column_config.DatetimeColumn(
        format='h:mm:ss', disabled=True
    )


TIME_CONFIG = st.column_config.TimeColumn(
    format='H:mm:ss', disabled=True
)
SPLIT_CONFIG = st.column_config.DatetimeColumn(
    format='m:ss.S', disabled=True
)


def time_config():
    return st.column_config.DatetimeColumn(
        format='H:mm:ss', disabled=True
    )


def garmin_activities_app(garmin_client, cols=None):
    print(f"Hello {garmin_client.full_name}")

    cols = cols or st.columns((1, 3, 3, 3))
    with cols[0]:
        st.image(
            garmin_client.garth.profile['profileImageUrlMedium'])
        st.write(f"Hello {garmin_client.full_name}")
    with cols[1]:
        limit = st.number_input(
            "How many garmin activities to load, "
            "set to 0 if selecting date range",
            value=1,
            min_value=0,
            step=1
        )
    with cols[2]:
        date1 = st.date_input(
            "Select Date",
            key="Garmin Select Date",
            value=pd.Timestamp.today() + pd.Timedelta("1d"),
            format='YYYY-MM-DD'
        )
    with cols[3]:
        date2 = st.date_input(
            "Range",
            key="Garmin Range",
            value=pd.Timestamp.today() - pd.Timedelta("7d"),
            format='YYYY-MM-DD'
        )
    activities = get_garmin_activities(
        garmin_client.username, limit, date1, date2)
    if len(activities):
        if activities is not None:
            st.divider()
            activity_hrs = get_garmin_activities_hr(
                garmin_client.username, activities.activityId)
            activities = activities.join(
                activity_hrs.droplevel(1, axis=1).add_prefix("TimeInZone"),
                on='activityId'
            )
            column_order = [
                c for c in ACTIVITY_FILTER_COLUMNS
                if c in activities.columns
            ]
            column_config = {
                "startTime": st.column_config.TimeColumn(format="h:mm a"),
                "duration": TIME_CONFIG,
                "movingDuration": TIME_CONFIG,
                "averageSplit": SPLIT_CONFIG,
                'TimeInZone1': TIME_CONFIG,
                'TimeInZone2': TIME_CONFIG,
                'TimeInZone3': TIME_CONFIG,
                'TimeInZone4': TIME_CONFIG,
                'TimeInZone5': TIME_CONFIG,
            }

            sel_activities = inputs.filter_dataframe(
                activities,
                select_all=False,
                select_first=True,
                key='garmin_activities',
                column_order=column_order,
                column_config=column_config,
                disabled=activities.columns.difference(['select', 'activity']),
                modification_container=st.popover("Filter Activities"),
            )

            with st.spinner("Downloading Activities"):
                garmin_data = {
                    activity.activity: load_garmin_fit(
                        garmin_client.username, activity.activityId
                    )
                    for _, activity in sel_activities.iterrows()
                }

            if st.toggle("Download gpx data"):
                download_gpx_files(garmin_client, sel_activities)
            if st.toggle("Download fit file"):
                download_fit_files(garmin_client, sel_activities)
            return garmin_data


@st.fragment
def download_fit_files(client, activities):
    for _, activity in activities.iterrows():
        fit = download_garmin_fit(
            client.username, activity.activityId
        )
        file_name = utils.safe_name(activity.activity)
        st.download_button(
            f":inbox_tray: Download: {file_name}.fit",
            fit,
            # type='primary',
            # mime="text/plain",
            file_name=f"{file_name}.fit",
        )


@st.fragment
def download_gpx_files(client, activities):
    for _, activity in activities.iterrows():
        gpx = download_garmin_gpx(
            client.username, activity.activityId
        )
        file_name = utils.safe_name(activity.activity)
        st.download_button(
            f":inbox_tray: Download: {file_name}.gpx",
            gpx,
            # type='primary',
            mime="text/plain",
            file_name=f"{file_name}.gpx",
        )


@st.fragment
def garmin_stats_app(garmin_client):
    cols = st.columns(2)
    with cols[0]:
        date1 = st.date_input(
            "Select Date",
            key="Garmin Stats Select Date",
            value=pd.Timestamp.today(),
            format='YYYY-MM-DD'
        )
    with cols[1]:
        date2 = st.date_input(
            "Range",
            key="Garmin Stats Range",
            value=pd.Timestamp.today() - pd.Timedelta("0d"),
            format='YYYY-MM-DD'
        )

    stats = get_garmin_sleep_stats(
        garmin_client.username, date1, date2
    ).sort_values("day", ascending=False)
    st.dataframe(
        stats,
        column_config={
            "day": st.column_config.DateColumn(),
            "sleepTime": st.column_config.TimeColumn(format="H:mm"),
            "sleepStartTimestampLocal": st.column_config.TimeColumn(format="H:mm a"),
            "sleepEndTimestampLocal": st.column_config.TimeColumn(format="H:mm a"),
            'deepSleep': st.column_config.TimeColumn(format="H:mm"),
            'lightSleep': st.column_config.TimeColumn(format="H:mm"),
            'remSleep': st.column_config.TimeColumn(format="H:mm"),
            'awakeSleep': st.column_config.TimeColumn(format="H:mm"),
        }
    )
    if st.toggle("Plot stats"):
        plot_stats(stats)


@st.fragment
def plot_stats(health_stats):
    health_stats = health_stats.reset_index()
    st.divider()
    with st.popover("Figure settings"):
        height = st.number_input(
            "Set profile figure height",
            100, None, 600, step=50,
            key='plot_garmin_stats_height',
        )

    options = health_stats.columns
    col0, col1, col2 = st.columns((1, 2, 2))
    with col0:
        x = st.selectbox(
            "Set x-axis",
            key='garmin_stats_x',
            options=options,
            index=0,
        )
    with col1:
        left_axis = st.multiselect(
            "Plot on left axis",
            key="garmin_stats_plotleft",
            default=['restingHeartRate'],
            options=options,
        )
    with col2:
        right_axis = st.multiselect(
            "Plot on right axis",
            key="garmin_stats_plotright",
            default=[],
            options=options,
        )

    fig = app.go.Figure()
    for c in left_axis:
        fig = app.scatter(
            health_stats, x, c, fig=fig
        )
    for c2 in right_axis:
        fig = app.scatter(
            health_stats, x, c2, fig=fig, yaxis='y2'
        )

    fig.update_layout(
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            title="+".join(left_axis)
        ),
        yaxis2=dict(
            title=" + ".join(right_axis)
        ),
        xaxis=dict(
            title=x
        )
    )
    st.plotly_chart(fig, use_container_width=True)
