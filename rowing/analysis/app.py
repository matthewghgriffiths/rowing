import streamlit as st
import io
import zipfile
import logging
import datetime
from itertools import count, cycle

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from rowing.analysis import splits, files, geodesy, telemetry, static
from rowing import utils
from rowing.app import threads, inputs


logger = logging.getLogger(__name__)

color_discrete_sequence = [
    '#636efa',
    '#EF553B',
    '#00cc96',
    '#ab63fa',
    '#FFA15A',
    '#19d3f3',
    '#FF6692',
    '#B6E880',
    '#FF97FF',
    '#FECB52'
]

DEFAULT_REPORT = {
    "report_0": {
        "select plot": "Heatmap"
    },
    "report_1": {
        # "pace boat time input": None,
        "select piece": "Pace Boat",
        "select plot": "Piece profile"
    },
    "report_2": {
        "select piece": "AvgBoatSpeed",
        "select plot": "Piece profile"
    },
    "report_3": {
        "select piece": "Rating",
        "select plot": "Piece profile"
    },
    "report_4": {
        "select piece": "Rower Swivel Power",
        "select plot": "Piece profile"
    },
    "report_5": {
        "select piece": "MinAngle",
        "select plot": "Piece profile"
    },
    "report_6": {
        "select piece": "MaxAngle",
        "select plot": "Piece profile"
    },
    "report_7": {
        "select piece": "Length",
        "select plot": "Piece profile"
    },
    "report_8": {
        "select piece": "CatchSlip",
        "select plot": "Piece profile"
    },
    "report_9": {
        "select piece": "FinishSlip",
        "select plot": "Piece profile"
    },
    "report_10": {
        "select piece": "Effective",
        "select plot": "Piece profile"
    },
    "report_11": {
        "Select x-axis": "GateAngle",
        "Select y-axis": "GateForceX",
        "rower profile figure height": 600,
        "select plot": "Stroke profile",
        "select stroke": "Rower profile"
    },
    "report_12": {
        "Select x-axis": "GateAngle",
        "Select y-axis": "GateAngleVel",
        "rower profile figure height": 600,
        "select plot": "Stroke profile",
        "select stroke": "Rower profile"
    },
    "report_13": {
        "Select x-axis": "Normalized Time",
        "Select y-axis": "GateForceX",
        "rower profile figure height": 600,
        "select plot": "Stroke profile",
        "select stroke": "Rower profile"
    },
    "report_14": {
        "select plot": "Stroke profile",
        "select stroke": "Boat profile",
        "select_boat_facets": [
            "Speed",
            "Accel",
            "Roll Angle",
            "Pitch Angle",
            "Yaw Angle"
        ],
        "select_boat_height": 1000
    },
    "report_setup": {
        "figure_height": 1000,
        "nview": 15,
        "toggleother": False,
        "window": 10
    }
}


def outlier_range(data, quantiles=(0.05, 0.5, 0.9)):
    y0, y1, y2 = data.quantile(quantiles)
    r = (y2 - y0) * 0.1
    dt = max(y2 - y1, y1 - y0) * 1.1
    yrange = (max(y1 - dt, data.min() - r), min(y1 + dt, data.max() + r))
    return yrange


def scatter(data, x, y, fig=None, **kwargs):
    fig = fig or go.Figure()

    xdata = data[x]
    ydata = data[y]

    xaxis = "xaxis" + kwargs.get("xaxis", "x")[1:]
    yaxis = "yaxis" + kwargs.get("yaxis", "y")[1:]
    yaxis_layout = dict(title=dict(text=y))
    xaxis_layout = dict(title=dict(text=x))
    if yaxis != "yaxis":
        yaxis_layout.update(
            side='right',
            tickmode="sync",
            overlaying="y",
            autoshift=True,
            automargin=True,
        )

    if x_is_td := pd.api.types.is_timedelta64_dtype(xdata):
        xdata = xdata + pd.Timestamp(0)
        xaxis_layout.update(
            tickformat="%-M:%S",
            range=outlier_range(xdata)
        )
    if y_is_td := pd.api.types.is_timedelta64_dtype(ydata):
        ydata = ydata + pd.Timestamp(0)
        yaxis_layout.update(
            tickformat="%-M:%S",
            range=outlier_range(ydata)
        )
    if y_is_obj := pd.api.types.is_object_dtype(ydata):
        t, = ydata.map(type).mode()
        if t == datetime.time:
            t0 = pd.Timestamp(0)
            ydata = ydata.map(lambda t: pd.Timestamp(
                year=t0.year, month=t0.month, day=t0.day,
                hour=t.hour,
                minute=t.minute,
                second=t.second,
                microsecond=t.microsecond,
            ) if pd.notna(t) else pd.Timestamp(np.nan))
            yaxis_layout.update(
                tickformat="%HH:%MM"
            )

    kwargs.setdefault("name", y)
    if 'text' not in kwargs:
        kwargs['text'] = data.apply((
            ("%s={0.%s} " % (x, x))
            + ("%s={0.%s}" % (y, y))
        ).format, axis=1),

    fig.add_trace(
        go.Scatter(
            x=xdata,
            y=ydata,
            **kwargs,
        )
    )
    fig.update_layout({
        yaxis: yaxis_layout,
        xaxis: xaxis_layout,
    })

    return fig


@ st.cache_data
def parse_gpx(file):
    return files.parse_gpx_data(files.gpxpy.parse(file))


def download_csv(
    file_name, df, label=":inbox_tray: Download data as csv", csv_kws=None, **kwargs
):
    st.download_button(
        label=label,
        file_name=file_name,
        data=df.to_csv(**(csv_kws or {})).encode("utf-8"),
        mime="text/csv",
        **kwargs,
    )


@ st.cache_data
def parse_telemetry_text(uploaded_files, use_names=True, sep='\t'):
    uploaded_data = {
        file.name.rsplit(".", 1)[0]: file.read().decode("utf-8")
        for file in uploaded_files
    }
    data, errs = utils.map_concurrent(
        telemetry.parse_powerline_text_data,
        uploaded_data,
        singleton=True,
        use_names=use_names,
        sep=sep
    )
    if errs:
        for k, err in errs.items():
            raise err
        logging.error(errs)

    return data


@st.cache_data
def parse_telemetry_files(uploaded_files, use_names=True):
    uploaded_data = {
        file.name.rsplit(".", 1)[0]: file for file in uploaded_files
    }
    data, errs = utils.map_concurrent(
        parse_file,
        uploaded_data,
        singleton=True,
        use_names=use_names,
    )
    if errs:
        for k, err in errs.items():
            raise err
        logging.error(errs)

    return data


@st.cache_data
def parse_file(file, use_names=True):
    filename, ending = file.name.rsplit(".", 1)
    ending = ending.lower()
    if ending == 'csv':
        return parse_text_data(file, use_names=use_names, sep=',')
    elif ending in {'xlsx', 'xls'}:
        return parse_excel(file, use_names=use_names)
    elif ending == 'zip':
        return telemetry.load_zipfile(file)
    return parse_text_data(file, use_names=use_names, sep='\t')


@st.cache_data
def parse_text_data(file, use_names=True, sep='\t'):
    return telemetry.parse_powerline_text_data(
        file.read().decode("utf-8"),
        use_names=use_names,
        sep=sep
    )


@st.cache_data
def parse_excel(file, use_names=True):
    data = pd.read_excel(file, header=None)
    return telemetry.parse_powerline_excel(data, use_names=use_names)


@ st.cache_data
def parse_telemetry_excel(uploaded_files, use_names=True):
    uploaded_data = {
        file.name.rsplit(".", 1)[0]: file
        # file.read().decode()
        for file in uploaded_files
    }
    data, errs = utils.map_concurrent(
        parse_excel,
        uploaded_data,
        singleton=True,
        use_names=use_names,
    )
    if errs:
        logging.error(errs)

    return data


@ st.cache_data
def parse_telemetry_zip(uploaded_files):
    telem_data = {}
    for file in uploaded_files:
        telem_data.update(
            telemetry.load_zipfile(file)
        )
        # with zipfile.ZipFile(file) as z:
        #     for f in z.filelist:
        #         name, key = f.filename.removesuffix(
        #             ".parquet").split("/")
        #         data = pd.read_parquet(
        #             z.open(f.filename)
        #         )
        #         if data.columns.str.contains("\(").any():
        #             data.columns = data.columns.map(ast.literal_eval)
        #         telem_data.setdefault(name, {})[key] = data

    return telem_data


@ st.cache_data
def get_crossing_times(gpx_data, locations=None, thresh=0.5):
    crossing_times, errors = utils.map_concurrent(
        splits.find_all_crossing_times,
        gpx_data,
        singleton=True,
        locations=locations,
        thresh=thresh,
    )
    if errors:
        logging.error(errors)

    return {
        k: d for k, d in crossing_times.items() if not d.empty
    }


@ st.cache_data
def get_location_timings(gpx_data, locations=None, thresh=0.5):
    location_timings, errors = utils.map_concurrent(
        splits.get_location_timings, gpx_data,
        singleton=True,
        locations=locations,
        thresh=thresh,
    )
    if errors:
        logging.error(errors)
    return location_timings


@ st.cache_data
def get_fastest_times(gpx_data):
    best_times, errors = utils.map_concurrent(
        splits.find_all_best_times,
        gpx_data, singleton=True,
    )
    if errors:
        logging.error(errors)
    return best_times


def select_pieces(all_crossing_times):
    if all_crossing_times.empty:
        return

    piece_dates = np.sort(all_crossing_times.dt.date.unique())
    cols = st.columns(4)
    with cols[0]:
        select_dates = st.multiselect(
            "select piece dates",
            piece_dates,
            piece_dates,
        )
        select_dates = pd.to_datetime(select_dates).date

    sel_times = all_crossing_times[
        all_crossing_times.dt.date.isin(select_dates)
    ].sort_index(level=(0, 4)).droplevel("location")

    longest_leg = sel_times.loc[
        sel_times.groupby(level=[0, 1]).size().idxmax()]
    leg_landmarks = longest_leg.index.get_level_values(0)
    other_landmarks = sel_times.index.get_level_values(
        'landmark').difference(leg_landmarks)
    landmarks = leg_landmarks.append(other_landmarks)

    if not len(leg_landmarks):
        return

    start, end = map(
        int, landmarks.get_indexer(
            [leg_landmarks[0], leg_landmarks[-1]]
        )
    )
    with cols[1]:
        start_landmark = st.selectbox(
            "select start landmark",
            landmarks,
            index=start,
        )
    with cols[2]:
        finish_landmark = st.selectbox(
            "select finish landmark",
            landmarks,
            index=end,
        )
    with cols[3]:
        intervals = st.number_input(
            "Enter distance intervals (m)",
            min_value=10, max_value=2000, value=None, step=10,
        )

    piece_data = splits.get_piece_times(
        sel_times, start_landmark, finish_landmark)

    if piece_data:
        return {
            "piece_data": piece_data,
            "start_landmark": start_landmark,
            "finish_landmark": finish_landmark,
            "intervals": intervals,
        }


def show_piece_data(piece_data, tabs=None):
    tabs = tabs or st.tabs(list(piece_data))
    for tab, (key, data) in zip(tabs, piece_data.items()):
        with tab:
            data = data.copy()
            for c, col in data.items():
                if pd.api.types.is_timedelta64_dtype(col.dtype):
                    data[c] = col.map(utils.format_timedelta)

            st.dataframe(data.reset_index(), hide_index=True)


PACE_DIST_COL = 'Distance ahead of Pace Boat (m)'
PACE_TIME_COL = 'Time ahead of Pace Boat (s)'


def align_pieces(gps_data, piece_data, landmark_distances=None, pace_boat_time=None, pieces=None):
    pace_boat_time = pd.Timedelta(
        pace_boat_time or piece_data['Elapsed Time'].iloc[:, -1].min(),
    ).total_seconds()
    if landmark_distances is None:
        landmark_distances = piece_data['Distance Travelled'].mean()[
            piece_data['Total Distance'].columns]

    pace_boat_kms = landmark_distances.max() / pace_boat_time
    pieces = piece_data['Total Distance'].index if pieces is None else pieces

    aligned_data = {}
    start_times = piece_data['Timestamp'].min(1)

    for piece in pieces:
        distances = piece_data['Total Distance'].loc[piece]
        name = piece[1]
        gps = gps_data[name]

        gps_adj = gps.copy()
        gps_adj['distance'] = np.interp(
            gps_adj.distance,
            distances,
            landmark_distances,
            left=np.nan, right=np.nan
        )
        gps_adj = gps_adj.dropna(
            subset='distance')
        gps_adj['timeElapsed'] = gps_adj['time'] - start_times.loc[piece]
        gps_adj[PACE_DIST_COL] = 1000 * (
            gps_adj.distance - gps_adj.timeElapsed.dt.total_seconds() * pace_boat_kms)
        gps_adj[PACE_TIME_COL] = (
            gps_adj.distance / pace_boat_kms - gps_adj.timeElapsed.dt.total_seconds())
        aligned_data[piece] = gps_adj.set_index('distance')

    if aligned_data:
        return pd.concat(
            aligned_data,
            names=piece_data['Timestamp'].index.names
        )
    return pd.DataFrame([])


def _align_pieces(piece_data, start_landmark, finish_landmark, gps_data, resolution=0.005):
    piece_distances = piece_data['Total Distance']
    piece_timestamps = piece_data['Timestamp']
    landmark_distances = piece_data['Distance Travelled'].mean()[
        piece_distances.columns]
    dists = np.arange(0, landmark_distances.max(), resolution)

    piece_gps_data = {}
    for piece in piece_distances.index:
        positions = gps_data[piece[1]]
        piece_gps_data[piece] = splits.get_piece_gps_data(
            positions,
            piece_distances.loc[piece],
            piece_timestamps.loc[piece],
            start_landmark,
            finish_landmark,
            landmark_distances
        )

    piece_compare_gps = pd.concat({
        piece: sel_data.set_index(
            "Distance Travelled"
        ).apply(utils.interpolate_series, index=dists)
        for piece, sel_data in piece_gps_data.items()
    }, axis=1, names=piece_distances.index.names
    ).rename_axis(index='distance')

    return piece_compare_gps


def upload_landmarks(landmarks):
    uploaded = st.file_uploader(
        "Upload landmarks csv",
        accept_multiple_files=False
    )
    if uploaded:
        uploaded_landmarks = pd.read_csv(uploaded)
        st.write("Uploaded Landmarks")
        st.dataframe(uploaded_landmarks, hide_index=True)
        landmarks = pd.concat(
            [uploaded_landmarks, landmarks]
        ).drop_duplicates().reset_index(drop=True)

    input = st.text_area(
        "Paste comma separated values in here, e.g. "
        "`tideway,boat_race_finish,51.4719098,-0.269101,122`"
    )
    if input:
        st.write("Entered Landmarks")
        new_landmarks = pd.read_csv(
            io.StringIO(input),
            header=None,
            names=landmarks.columns
        )
        st.dataframe(new_landmarks)
        landmarks = pd.concat(
            [new_landmarks, landmarks]
        ).drop_duplicates().reset_index(drop=True)

    return landmarks


def add_piece_landmarks(gps_data, new_landmarks=None):
    new_landmarks = new_landmarks or {}
    new_locations = pd.DataFrame()
    if not gps_data:
        st.write("no gps data loaded")
        return new_locations, new_landmarks

    cols = st.columns((1, 5))
    with cols[0]:
        st.markdown("<br>", unsafe_allow_html=True)
        count = st.empty()
        n_pick = st.number_input(
            "Number of landmarks",
            min_value=0,
            value=0,
            step=1,
            key='npick_distance'
        )
        with count:
            st.write(
                f"\nSetting {n_pick} landmarks from pieces")

    with cols[1]:
        for i in range(n_pick):
            cols = st.columns(3)
            with cols[0]:
                name = st.selectbox(
                    "Pick piece",
                    options=list(gps_data.keys()),
                    key=f"Pick piece {i}",
                )
                data = gps_data[name]
            with cols[1]:
                dist = st.number_input(
                    "Select distance (km)",
                    min_value=data.distance.min(),
                    max_value=data.distance.max(),
                    step=0.1,
                    format="%.3f",
                    key=f"Pick piece distance {i}",
                )
            with cols[2]:
                landmark = st.text_input(
                    "Enter landmark name",
                    value=f"{name} {dist:.3f} km",
                    key=f"Pick piece landmark {i}",
                )
            position = data.set_index("distance")[
                ['latitude', 'longitude', 'bearing']
            ].apply(
                utils.interpolate_series, index=[dist]
            ).rename_axis(index='distance')
            new_landmarks[name, landmark] = position

    if new_landmarks:
        new_locations = pd.concat(
            new_landmarks, names=['location', 'landmark'], axis=0
        ).reset_index("distance", drop=True).reset_index()

        download_csv(
            "piece_landmarks.csv",
            new_locations,
            ':inbox_tray: download piece landmarks as csv',
            csv_kws=dict(index=False),
        )

    return new_locations, new_landmarks


def edit_landmarks(landmarks):
    col1, col2 = st.columns(2)
    locations = landmarks.location.unique()
    with col1:
        sel_locations = st.multiselect(
            "filter", locations, default=locations
        )
        edited_landmarks = st.data_editor(
            landmarks[
                landmarks.location.isin(sel_locations)
            ],
            hide_index=True,
            num_rows="dynamic"
        )
    with col2:
        st.write(
            """
            Add new landmarks by entering at the bottom.

            Delete old landmarks by selecting the left column and pressing delete

            Hold shift to select multiple landmarks

            The current set of landmarks can be downloaded as a csv

            A custom landmarks can be uploaded as a csv which will be merged with the existing landmarks.
            This csv must match the format of the downloaded csv
            """)

    download_csv(
        "landmarks.csv",
        edited_landmarks,
        ':inbox_tray: download set landmarks as csv',
        csv_kws=dict(index=False),
    )

    return edited_landmarks


def set_landmarks(gps_data=None, landmarks=None, title=True):
    tab0, tab1, tab2 = st.tabs([
        "From Pieces",
        "Edit Landmarks",
        "Upload Landmarks",  # "Map of Landmarks"
    ])

    if landmarks is None:
        landmarks = splits.load_location_landmarks().reset_index()

    if title:
        landmarks['landmark'] = landmarks['landmark'].str.replace(
            "_", " "
        ).str.title().str.replace(
            r"([0-9][A-Z])",
            lambda m: m.group(0).lower(),
            regex=True
        )

    with tab0:
        new_locations, new_landmarks = add_piece_landmarks(gps_data)
        landmarks = pd.concat([new_locations, landmarks])

    with tab2:
        landmarks = upload_landmarks(landmarks)

    with tab1:
        col1, col2 = st.columns(2)
        locations = landmarks.location.unique()
        with col1:
            sel_locations = st.multiselect(
                "filter", locations, default=locations
            )
            set_landmarks = st.data_editor(
                landmarks[
                    landmarks.location.isin(sel_locations)
                ],
                hide_index=True,
                num_rows="dynamic"
            )
        with col2:
            st.write(
                """
                Add new landmarks by entering at the bottom.

                Delete old landmarks by selecting the left column and pressing delete

                Hold shift to select multiple landmarks

                The current set of landmarks can be downloaded as a csv

                A custom landmarks can be uploaded as a csv which will be merged with the existing landmarks.
                This csv must match the format of the downloaded csv
                """)

        download_csv(
            "landmarks.csv",
            set_landmarks,
            ':inbox_tray: download set landmarks as csv',
            csv_kws=dict(index=False),
        )

    with tab2:
        download_csv(
            "landmarks.csv",
            set_landmarks,
            ':inbox_tray: download landmarks as csv',
            csv_kws=dict(index=False),
        )

    with tab0:
        st.subheader("Map of Landmarks")
        cols = st.columns([5, 2])
        with cols[0]:
            map_style = st.selectbox(
                "map style",
                ["open-street-map", "carto-positron", "carto-darkmatter"],
                key='landmark map style'

            )
        with cols[1]:
            height = st.number_input(
                "Set figure height", 100, 3000, 600, step=50,
                key='landmark map height'
            )

        points = draw_gps_landmarks(
            gps_data, set_landmarks, new_landmarks, map_style=map_style, height=height)

    return set_landmarks


# Terrible hacks to fix plotly_mapbox_events...
MAPBOX_WIDTHS = cycle(['100%', '100.1%'])
MAPBOX_SUFFIXES = cycle(['', ' '])


def set_landmarks(gps_data=None, landmarks=None, title=True):
    tab0, tab1, tab2 = tabs = st.tabs([
        "From Pieces",
        "Edit Landmarks",
        "Upload Landmarks",  # "Map of Landmarks"
    ])
    if landmarks is None:
        landmarks = splits.load_location_landmarks().reset_index()
        landmarks['original'] = True

    with tab0:
        st.subheader("Landmarks from Activities")
        st.markdown(
            """Click on a track on the map to add a custom landmark,
        Landmarks can be selecting the box on the right of the entry,
        the table can be directly edited to change the name/location.

        """)
        points, sel_landmarks = points_to_landmarks(get_points(gps_data))
        if not sel_landmarks.empty:
            landmarks = pd.concat([sel_landmarks, landmarks])

    with tab1:
        with st.container():
            landmarks = edit_landmarks(landmarks)

    with tab2:
        with st.container():
            landmarks = upload_landmarks(landmarks)

    st.divider()
    map = st.container()
    with st.popover("Settings"):
        map_style = st.selectbox(
            "map style",
            ["open-street-map", "carto-positron", "carto-darkmatter"],
            key='landmark map style'
        )
        height = st.number_input(
            "Set figure height", 100, None, 800, step=50,
            key='landmark map height'
        )

    fig = make_gps_landmarks_figure(
        gps_data, landmarks, map_style=map_style, height=height, suffix=next(MAPBOX_SUFFIXES))
    with map:
        st.subheader("Map")
        clickable_map(fig, height, next(MAPBOX_WIDTHS))
        update_points(points)

    download_csv(
        "landmarks.csv",
        landmarks,
        ':inbox_tray: download landmarks as csv',
        csv_kws=dict(index=False),
    )

    return landmarks


def points_to_landmarks(points):
    landmarks_cols = ['location', 'landmark',
                      'latitude', 'longitude', 'bearing']
    if points:
        tables = pd.DataFrame.from_records(points)
        tables['point'] = tables.index
        tables['location'] = tables['name']
        tables['landmark'] = tables.apply(
            "{0.distance:.2f}k".format, axis=1
        )
    else:
        tables = pd.DataFrame(
            [], columns=landmarks_cols + ['distance', 'point'])

    sel_landmarks = st.data_editor(
        tables, column_order=landmarks_cols + ['distance'], num_rows='dynamic')

    points = [points[i] for i in sel_landmarks.point.dropna()]
    sel_landmarks['original'] = False
    return points, sel_landmarks[landmarks_cols + ['original']]


def get_points(gps_data):
    points = st.session_state.get('points', [])
    if st.session_state.get('new_point'):
        st.session_state['new_point'] = False
        point = st.session_state['last_point']
        matched_point = match_point(point.copy(), gps_data)
        if "name" in matched_point:
            points.append(matched_point)

    return points


def update_points(points):
    if st.session_state.get('new_point'):
        st.rerun()
    else:
        st.session_state['points'] = points


def match_point(point, gps_data):
    i = point['pointIndex']
    for name, gps in gps_data.items():
        if len(gps) > i:
            track = gps.iloc[i].to_dict()
            match = (
                (point['lat'] == track['latitude'])
                and (point['lon'] == track['longitude'])
            )
            if match:
                point['name'] = name
                point.update(track)
    return point


@ st.fragment
def clickable_map(fig, height=800, width='100%'):
    from streamlit_plotly_mapbox_events import plotly_mapbox_events
    # fig.config(responsive=True)
    points, *_ = plotly_mapbox_events(
        fig,
        override_width=width,
        override_height=height,
        key='LandmarksMap'
    )
    if points:
        point, = points
        last_point = st.session_state.get('last_point', None)
        if point != last_point:
            st.session_state['last_point'] = point
            st.session_state['new_point'] = True
            st.rerun()


@ st.cache_data
def make_gps_landmarks_figure(gps_data, landmarks, map_style='open-street-map', height=600, suffix=' '):
    fig = go.Figure()
    color_cycle = cycle(color_discrete_sequence)
    landmark_color = next(color_cycle)

    if gps_data:
        for name, data in gps_data.items():
            fig.add_trace(go.Scattermapbox(
                lon=data.longitude,
                lat=data.latitude,
                mode='lines',
                name=name + suffix,
                line_color=next(color_cycle),
                legendgroup='Activities',
                legendgrouptitle_text='Activities',
            ))

    fig.add_trace(go.Scattermapbox(
        lon=landmarks.longitude,
        lat=landmarks.latitude,
        # customdata = set_landmarks,
        mode='markers+text',
        name='Landmarks' + suffix,
        text=landmarks.landmark,
        cluster=dict(
            enabled=True,
            maxzoom=5,
            step=1,
            size=20,
        ),
        marker={
            'size': 5,
            'color': landmark_color,
        },
        legendgroup='Landmarks',
        legendgrouptitle_text='Landmarks',
        textposition='bottom right',
    ))

    first = True
    landmark_kws = dict(
        name="Bearings" + suffix,
        marker={"color": landmark_color},
        legendgroup='Landmarks',
        legendgrouptitle_text='Landmarks',
    )
    set_kws = dict(
        showlegend=True,
        legendgroup='Set Landmarks',
        legendgrouptitle_text='Set Landmarks',
    )
    for i, landmark in landmarks.iterrows():
        arrow = geodesy.make_arrow_base(landmark, 0.25, 0.1, 20)
        if landmark.original:
            landmark_kws["showlegend"] = first
            first = False
            kws = landmark_kws
        else:
            set_kws.update(
                name=f"{landmark.location} @ {landmark.landmark}" + suffix,
                marker={"color": next(color_cycle)},
            )
            kws = set_kws

        trace = go.Scattermapbox(
            lon=arrow.longitude,
            lat=arrow.latitude,
            mode='lines',
            fill='toself',
            hovertext=f"{landmark.landmark} bearing={landmark.bearing:.1f}",
            line=dict(width=3,),
            textposition='bottom right',
            **kws
        )
        fig.add_trace(trace)

    lon = landmarks.longitude.mean()
    lat = landmarks.latitude.mean()
    zoom = 9

    if gps_data:
        positions = pd.concat(gps_data)
        lon = positions.longitude.mean()
        lat = positions.latitude.mean()
        scale = max(np.ptp(positions.longitude), np.ptp(positions.latitude))
        zoom = 8.55 - 3.25 * np.log10(scale)

    fig.update_layout(
        {"uirevision": True},
        mapbox={
            'style': map_style,
            'center': {'lon': lon, 'lat': lat},
            'zoom': zoom
        },
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            itemsizing='constant',
            # yref='paper',
            # entrywidth=50,
            # autosize=True
        ),
        height=height,
        overwrite=True,
        autosize=True,
        margin=dict(
            b=0, l=0, r=0, t=0,
            pad=10,
            autoexpand=True,
        )
        # template=pio.templates.default,
    )
    return fig


def draw_gps_landmarks(gps_data, set_landmarks, new_landmarks, map_style='open-street-map', height=600):
    fig = make_gps_landmarks_figure(
        gps_data, set_landmarks, new_landmarks, map_style=map_style, height=height)
    state = st.plotly_chart(
        fig,
        use_container_width=True,
        selection_mode=['points'],
    )
    return state


@ st.fragment
def draw_gps_data(gps_data, locations, index=None):
    fig = make_gps_figure(gps_data, locations, index)
    st.plotly_chart(fig, use_container_width=True)
    return fig


gps_figure_count = count()


def make_gps_figure(gps_data, locations, index=None):
    index = index or next(gps_figure_count)
    cols = st.columns([5, 2])
    with cols[0]:
        map_style = st.selectbox(
            "map style",
            ["open-street-map", "carto-positron", "carto-darkmatter"],
            key=f'gps map style {index}'
        )
    with cols[1]:
        height = st.number_input(
            "Set figure height", 100, 2000, 600,
            key=f'gps map height {index}'
        )

    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lon=locations.longitude,
        lat=locations.latitude,
        # hoverinfo = landmark_locs.index,
        mode='markers+text',
        name='Landmarks',
        text=locations.index.get_level_values("landmark"),
        marker={
            'size': 20,
            # 'symbol': "airfield",
            # 'icon': dict(iconUrl="https://api.iconify.design/maki-city-15.svg"),
        },
        textposition='bottom right',
        showlegend=True,
    ))
    data = locations
    for name, data in gps_data.items():
        fig.add_trace(go.Scattermapbox(
            lon=data.longitude,
            lat=data.latitude,
            mode='lines',
            name=name,
            showlegend=True,
        ))
    fig.update_layout(
        mapbox={
            'style': map_style,
            'center': {
                'lon': data.longitude.mean(),
                'lat': data.latitude.mean(),
            },
            'zoom': 10
        },
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        showlegend=True,
        height=height,
    )
    fig.update_traces(showlegend=True)
    return fig


@st.cache_data
def make_stroke_profiles(telemetry_data, piece_data, nres=101):
    profiles = {}
    boat_profiles = {}
    crew_profiles = {}
    for piece, piece_times in piece_data['Timestamp'].iterrows():
        name, leg = piece[1:3]
        profile = telemetry_data[name]['Periodic'].sort_index(axis=1)
        start_time = piece_times.min()
        finish_time = piece_times.max()
        piece_profile = profile[
            profile.Time.dt.tz_localize(None).between(start_time, finish_time)
        ].set_index('Time').dropna(axis=1, how='all')

        profiles[name, leg] = profile = telemetry.norm_stroke_profile(
            piece_profile, nres)
        gate_angle = profile.GateAngle
        gate_angle0 = gate_angle - gate_angle.values.mean(0, keepdims=True)
        for (pos, side), angle0 in gate_angle0.items():
            profile["GateAngle0", pos, side] = angle0

        mean_profile = profile.groupby(
            level=1
        ).mean().reset_index().rename(
            {"": "Boat"}, axis=1, level=1
        )
        boat_profiles[name, leg] = boat_profile = mean_profile.xs(
            "Boat", axis=1, level=1).droplevel(axis=1, level=1)

        profile = mean_profile[
            mean_profile.columns.levels[0].difference(
                boat_profile.columns)
        ].rename_axis(
            columns=("Measurement", "Position", 'Side')
        ).stack(level=[1, 2], future_stack=True).reset_index(
            ["Position", 'Side']
        ).rename_axis(
            index="Normalized Time"
        ).reset_index()
        crew_profiles[name, leg] = profile

    crew_profile = pd.concat(
        crew_profiles, names=['name', 'leg']
    ).reset_index(['name', 'leg'])
    crew_profile['Rower'] = (
        # crew_profile.Position + "|" + crew_profile.File
        crew_profile.name + "|" + crew_profile.Position
    )

    return {
        "profiles": profiles,
        "boat_profiles": boat_profiles,
        "crew_profiles": crew_profiles,
        "crew_profile": crew_profile,
    }


@ st.cache_data
def make_telemetry_figure(piece_power, col, name, start_time, epoch_times):
    if col == 'Work PC':
        WorkPC_cols = [
            'Work PC Q1', 'Work PC Q2', 'Work PC Q3', 'Work PC Q4']
        pc_work = piece_power.set_index("Time")[
            WorkPC_cols
        ].stack([0, 1]).rename("PC").reset_index()
        pc_work['Elapsed'] = (
            (pc_work['Time'] - start_time) + pd.Timestamp(0)
        ).dt.tz_localize(None)
        fig = px.area(
            pc_work,
            x="Elapsed",
            y='PC',
            facet_col='Position',
            facet_col_wrap=4,
            color='Measurement',
            title=name,
            color_discrete_sequence=color_discrete_sequence,
            template="plotly_white",
        )
    else:
        plot_data = piece_power.stack(1)[
            ['Time', col]
        ]
        plot_data['Time'] = plot_data['Time'].ffill()
        plot_data['Elapsed'] = (
            (plot_data['Time'] - start_time) + pd.Timestamp(0)
        ).dt.tz_localize(None)
        plot_data = plot_data.dropna().reset_index()
        fig = px.line(
            plot_data,
            x='Elapsed',
            y=col,
            color='Position',
            title=name,
            template="plotly_white",
            color_discrete_sequence=color_discrete_sequence,
        )

    for landmark, epoch in epoch_times.items():
        fig.add_vline(
            x=int((epoch - 3600) * 1000),
            annotation_text=landmark,
            annotation=dict(
                textangle=-90
            )
        )

    fig.update_xaxes(
        tickformat="%M:%S",
        dtick=60*1000,
        showgrid=True,
        griddash='solid',
    )
    fig.update_traces(visible=True)
    return fig


def plot_pace_boat(piece_data, landmark_distances, gps_data, height=600, input_container=None, name='name', key=''):
    piece_distances = piece_data['Total Distance']
    piece_timestamps = piece_data['Timestamp']
    dists = np.arange(0, landmark_distances.max(), 0.005)

    start_landmark = landmark_distances.idxmin()
    finish_landmark = landmark_distances.idxmax()

    piece_gps_data = {}
    for piece in piece_distances.index:
        positions = gps_data[piece[1]]
        piece_gps_data[piece] = splits.get_piece_gps_data(
            positions,
            piece_distances.loc[piece],
            piece_timestamps.loc[piece],
            start_landmark,
            finish_landmark,
            landmark_distances
        )

    piece_compare_gps = pd.concat({
        piece: sel_data.set_index(
            "Distance Travelled"
        ).apply(utils.interpolate_series, index=dists)
        for piece, sel_data in piece_gps_data.items()
    }, axis=1, names=piece_distances.index.names
    ).rename_axis(
        index='distance'
    )
    boat_times = piece_compare_gps.xs('timeElapsed', level=-1, axis=1)
    pace_boat_finish = boat_times.iloc[-1].rename(
        "Pace boat time") + pd.Timestamp(0)
    pace_boat_finish[:] = pace_boat_finish.min()
    pace_boat_finish = pd.concat([
        boat_times.iloc[-1].rename("Finish time")
        + pd.Timestamp(0),
        pace_boat_finish,
    ], axis=1).reset_index()

    with input_container or st.container():
        cols = st.columns(2)
        with cols[0]:
            st.write("Set pace boat time")
        with cols[1]:
            st.text_input(
                "Set pace boat time",
                value=None,
                # step=1,
                # format="m:ss.S"
                placeholder="m:ss.S",
                key=key+"pace boat time input",
            )

    pace_boat_finish['Pace boat time'] -= pd.Timestamp(0)
    pace_boat_finish = pace_boat_finish.set_index(
        boat_times.columns.names)['Pace boat time']
    pace_boat_time = pd.DataFrame(
        pace_boat_finish.values[None, :] * dists[:, None] / dists[-1],
        index=boat_times.index, columns=pace_boat_finish.index
    )
    time_behind = (
        boat_times - pace_boat_time
    ).unstack().dt.total_seconds().rename("time behind pace boat (s)").reset_index()
    names = piece_names(time_behind, name).set_index([
        name, "leg"
    ])
    time_behind = time_behind.join(
        names.piece, on=[name, 'leg']
    )

    fig = px.line(
        time_behind,
        x='distance',
        y='time behind pace boat (s)',
        color="name",
        line_dash='leg',
        template="plotly_white",
    )

    for landmark, distance in landmark_distances.items():
        fig.add_vline(
            x=distance,
            annotation_text=landmark,
            annotation=dict(
                textangle=-90
            )
        )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=height)

    time_behind = time_behind.set_index(
        "distance"
    ).groupby(
        [name, "leg"]
    )['time behind pace boat (s)'].apply(
        utils.interpolate_series, index=landmark_distances
    ).unstack()

    return fig, time_behind


def piece_names(data, name='name', leg='leg'):
    pieces = data.groupby(
        [name, leg]
    ).size().rename("count").reset_index()[[name, leg]]
    pieces = pieces.join(
        pieces.groupby(name).size().rename("n_legs"),
        on=name
    )
    pieces['piece'] = pieces[name] + np.select(
        pieces.n_legs == 1,
        pieces[leg].apply("".format),
        pieces[leg].apply(" leg={}".format)
    )
    return pieces


# @st.cache_data
def make_telemetry_distance_figure(compare_power, landmark_distances, col, facet_col_wrap=4):
    n_legs = compare_power.groupby(
        ["name", "leg"]
    ).size().groupby(level=0).size()

    if col == 'Work PC':
        WorkPC_cols = [
            'Work PC Q1', 'Work PC Q2', 'Work PC Q3', 'Work PC Q4']
        pc_work = compare_power[
            ['name', 'leg', 'Distance', 'Position'] + WorkPC_cols
        ].copy()
        pc_work['piece'] = pc_work.name + np.select(
            n_legs.loc[pc_work.name] == 1,
            pc_work.leg.apply("".format),
            pc_work.leg.apply(" leg={}".format)
        )
        pc_work['R'] = pc_work['piece'].str.cat(
            pc_work.Position, sep="|"
        )
        pc_plot_work = pc_work.set_index(
            ["Distance", "R"]
        )[WorkPC_cols].stack().rename(col).reset_index()

        fig = px.area(
            pc_plot_work,
            x="Distance",
            y=col,
            facet_col='R',
            facet_col_wrap=facet_col_wrap,
            color='Measurement',
            facet_col_spacing=0.01,
            facet_row_spacing=0.02,
            template="plotly_white",
            # title=name,
            # color_discrete_sequence=app.color_discrete_sequence,
        )
    else:
        fig = go.Figure()
        for (file, leg), data in compare_power.groupby(["name", "leg"]):
            cols = data[[col]].columns
            pos_power = data.dropna(
                subset=cols, how='all'
            ).groupby(["Position", 'Side'])

            for (pos, side), pos_data in pos_power:
                name = f"{file} {side}" if n_legs[file] == 1 else f"{file} {side} {leg=}"
                for c in cols:
                    fig.add_trace(
                        go.Scatter(
                            x=pos_data["Distance"],
                            y=pos_data[c],
                            legendgroup=f"{file} {leg} {side}",
                            legendgrouptitle_text=name,
                            name=f"{pos}",
                            mode='lines',
                        )
                    )
        fig.update_layout(
            xaxis_title="Distance (km)",
            yaxis_title=col,
        )

    for landmark, distance in landmark_distances.items():
        fig.add_vline(
            x=distance,
            annotation_text=landmark,
            annotation=dict(
                textangle=-90
            )
        )
    return fig


@ st.cache_data
def make_telemetry_figures(telemetry_data, piece_data, window: int = 0, tab_names=None):
    if tab_names is None:
        tab_names = telemetry.FIELDS

    telemetry_figures = {}
    for piece, piece_times in piece_data['Timestamp'].iterrows():
        name = piece[1]
        power = telemetry_data[name]['power']
        if window:
            time_power = power.set_index("Time").sort_index()
            avg_power = time_power.rolling(
                pd.Timedelta(seconds=window)
            ).mean()
            power = avg_power.reset_index()

        # piece_times = piece_data['Timestamp'].xs(name, level=1).iloc[0]
        start_time = piece_times.min()
        finish_time = piece_times.max()
        piece_power = power[
            power.Time.between(start_time, finish_time)
        ]
        piece_power.columns.names = 'Measurement', 'Position'

        epoch_times = (
            (piece_times - start_time)  # + pd.Timestamp(0)
        ).dt.total_seconds()
        figures, errors = utils.map_concurrent(
            make_telemetry_figure,
            {
                col: (piece_power, col, name, start_time, epoch_times)
                for col in tab_names
            },
            # max_workers=1
            # progress_bar=None,
        )
        if errors:
            logger.warning(errors)
        for col, fig in figures.items():
            telemetry_figures[col, name] = fig

    return telemetry_figures


@ st.cache_data
def figures_to_zipfile(figures, file_type, **kwargs):
    zipdata = io.BytesIO()
    with zipfile.ZipFile(zipdata, 'w') as zipf:
        for name, fig in figures.items():
            if file_type == 'html':
                fig_data = fig.to_html(**kwargs)
            else:
                fig_data = fig.to_image(format=file_type, **kwargs)

            zipf.writestr(
                f"{name}.{file_type}", fig_data
            )

    zipdata.seek(0)
    return zipdata


def save_figure_html(
        figure,
        label='Download Figure',
        file_name='figure.html', include_plotlyjs=True, **kwargs):
    html_data = figure.to_html(
        include_plotlyjs=include_plotlyjs,
    )
    st.download_button(
        label=label,
        data=html_data,
        file_name=file_name,
        mime='text/html',
    )


def telemetry_to_zipfile(telemetry_data):
    zipdata = io.BytesIO()
    with zipfile.ZipFile(zipdata, 'w') as zipf:
        for name, piece_data in telemetry_data.items():
            for k, data in piece_data.items():
                if isinstance(data, pd.DataFrame):
                    save_data = data.copy()
                elif isinstance(data, pd.Series):
                    save_data = data.reset_index()

                for c, vals in save_data.items():
                    if pd.api.types.is_object_dtype(vals.dtype):
                        save_data[c] = vals.astype(str)

                with zipf.open(f"{name}/{k}.parquet", "w") as f:
                    save_data.to_parquet(f, index=False)

    zipdata.seek(0)
    return zipdata


def setup_plots(piece_rowers, state, default_height=600, key='', toggle=True, nview=False, cols=None, input_container=None):
    if not cols:
        with input_container or st.container():
            cols = st.columns((1, 1, 5))

    with cols[0]:
        all_plots = None
        if toggle:
            all_plots = st.toggle(
                'Make all plots',
                key=key + 'Make all plots')
        elif nview:
            all_plots = st.number_input(
                "Number of panels",
                min_value=0,
                value=0,
                step=1,
                key=key + 'nview'
            )

    show_rowers = None
    with cols[1]:
        toggle_athletes = st.toggle(
            "Filter athletes", key=key + "toggleother"
        )
        with cols[2]:
            if toggle_athletes:
                cols2 = st.columns((3, 2, 2))
                with cols2[0]:
                    show_rowers = st.multiselect(
                        "Select athletes to plot",
                        options=piece_rowers.map("|".join),
                        key=key + "show_athletes",
                    )
            else:
                cols2 = st.columns(2)

    with cols2[-2]:
        window = st.number_input(
            "Select window to average over (s), set to 0 to remove smoothing",
            value=10,
            min_value=0,
            step=5,
            key=key + 'window'
        )
    with cols2[-1]:
        height = st.number_input(
            "Set figures height",
            100, 3000, default_height, step=50,
            key=key+'figure_height',
        )
    return window, show_rowers, all_plots, height


def setup_plot_data(piece_information, window, show_rowers=None):
    piece_information['show_rowers'] = show_rowers

    piece_data = piece_information['piece_data']
    telemetry_data = piece_information['telemetry_data']
    gps_data = piece_information['gps_data']

    piece_distances = piece_data['Total Distance']
    landmark_distances = piece_data['Distance Travelled'].mean()[
        piece_distances.columns
    ].sort_values()
    compare_power = telemetry.compare_piece_telemetry(
        telemetry_data, piece_data, gps_data, landmark_distances,
        window=int(window))
    piece_information['n_legs'] = compare_power.groupby(
        ["name", "leg"]
    ).size().groupby(level=0).size()
    piece_data_filter = piece_data

    if show_rowers:
        show_rowers = pd.MultiIndex.from_tuples([
            tuple(r.split("|", 2)) for r in show_rowers
        ], names=["Position", "name"])
        filter_rows = pd.MultiIndex.from_frame(
            compare_power[["Position", "name"]]
        ).isin(show_rowers)
        compare_power = compare_power[filter_rows]
        piece_data_filter = {
            k: data.reindex(show_rowers.swaplevel(0, 1))
            if data.index.nlevels == 2
            and len(data.index.intersection(show_rowers.swaplevel(0, 1)))
            else data
            for k, data in piece_data.items()
        }

    piece_information['compare_power'] = compare_power
    piece_information['landmark_distances'] = landmark_distances
    piece_information['piece_data_filter'] = piece_data_filter

    return piece_information


def plot_piece_data(piece_information, show_rowers, all_plots, height):
    telemetry_figures = {}
    if not piece_information:
        return telemetry_figures

    start_landmark = piece_information['start_landmark']
    finish_landmark = piece_information['finish_landmark']
    compare_power = piece_information['compare_power']
    landmark_distances = piece_information['landmark_distances']
    piece_data_filter = piece_information['piece_data_filter']

    tab_names = ["Pace Boat"] + list(telemetry.FIELDS)
    telem_tabs = dict(zip(tab_names, st.tabs(tab_names)))
    for col, tab in telem_tabs.items():
        with tab:
            cols = st.columns((1, 7))

            with cols[0]:
                on = st.toggle('Make plot', value=all_plots,
                               key=col + ' make plot')

            _height = height
            facet_col_wrap = 4
            if col == "Pace Boat" and on:
                fig, time_behind = plot_pace_boat(
                    piece_information['piece_data'],
                    piece_information['landmark_distances'],
                    piece_information['gps_data'],
                    height=_height,
                    input_container=cols[1],
                )
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("Time behind pace boat")
                st.dataframe(time_behind)
                telemetry_figures[
                    col, f"{start_landmark} to {finish_landmark}"] = fig

            elif on:
                if col == 'Work PC':
                    with cols[1]:
                        cols2 = st.columns(2)

                    n_plots = len(
                        compare_power[['name', 'leg', 'Position']].value_counts())
                    if show_rowers:
                        n_plots = len(show_rowers)

                    with cols2[0]:
                        facet_col_wrap = st.number_input(
                            "Select number of columns",
                            value=4, min_value=1, step=1,
                        )
                        n_rows = np.ceil(n_plots / facet_col_wrap)
                    with cols2[1]:
                        _height = st.number_input(
                            "Set Work PC figure height",
                            min_value=100,
                            # 3000,
                            value=int(height * n_rows // 2),
                            step=50,
                        )

                fig = make_telemetry_distance_figure(
                    compare_power, landmark_distances, col,
                    facet_col_wrap=facet_col_wrap,
                )
                itemclick = 'toggle'
                itemdoubleclick = "toggleothers"
                groupclick = 'toggleitem'
                fig.update_layout(
                    title=f"{col}: {start_landmark} to {finish_landmark}",
                    height=_height,
                    legend=dict(
                        itemclick=itemclick,
                        itemdoubleclick=itemdoubleclick,
                        groupclick=groupclick,
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
                st.write(
                    "Click on legend to toggle traces, "
                    "double click to select only one piece"
                )
                telemetry_figures[
                    col, f"{start_landmark} to {finish_landmark}"] = fig

            cols = st.columns(2)
            with cols[0]:
                interval_stats = piece_data_filter.get(
                    f"Interval {col}")
                if interval_stats is not None:
                    st.subheader("Interval Averages")
                    st.write(interval_stats)

            with cols[1]:
                interval_stats = piece_data_filter.get(
                    f"Average {col}")
                if interval_stats is not None:
                    st.subheader("Piece Averages")
                    st.write(interval_stats)

    return telemetry_figures


def plot_piece_data(piece_information, show_rowers, all_plots, height):
    piece_figures = {}
    piece_tables = {}

    tab_names = ["Pace Boat"] + list(telemetry.FIELDS)
    telem_tabs = dict(zip(tab_names, st.tabs(tab_names)))
    for col, tab in telem_tabs.items():
        with tab:
            cols = st.columns((1, 7))
            with cols[0]:
                on = st.toggle('Make plot', value=all_plots,
                               key=col + ' make plot')

            if on:
                figures, tables = plot_piece_col(
                    col, piece_information,
                    default_height=height,
                    key=col, input_container=cols[1]
                )
                for c, fig in figures.items():
                    st.plotly_chart(fig, use_container_width=True)
                    piece_figures['piece', c] = fig

                piece_tables.update(tables)
                for t, table in tables.items():
                    st.subheader(t)
                    st.dataframe(table, use_container_width=True)

    return piece_figures, piece_tables


def plot_rower_profiles(piece_information, default_height=600, key="rower_", input_container=None, cols=None):
    crew_profiles = piece_information['crew_profiles']

    if not cols:
        input_container = input_container or st.container()
        with input_container:
            cols = st.columns(3)

    with cols[0]:
        x = st.selectbox(
            "Select x-axis",
            ['GateAngle', 'Normalized Time', 'GateForceX',
                'GateAngleVel', "GateAngle0"],
            key=key+"Select x-axis",
        )
    with cols[1]:
        y = st.selectbox(
            "Select y-axis",
            ['GateForceX', 'GateAngle', 'GateAngleVel',
                "GateAngle0", 'Normalized Time'],
            key=key+"Select y-axis",
        )
    with cols[2]:
        height = st.number_input(
            "Set figure height",
            min_value=100,
            max_value=None,
            value=default_height,
            step=100,
            key=key+"rower profile figure height",
        )
    # with cols[3]:
    #     ymin = float(min(
    #         profile[y].min() for profile in crew_profiles.values()
    #     ))
    #     ymax = float(max(
    #         profile[y].max() for profile in crew_profiles.values()
    #     ))
    #     yr = float(ymax - ymin)
    #     yrange = st.slider(
    #         "Set y lims",
    #         ymin - yr/10, ymax + yr /
    #         10, (ymin - yr/10, ymax + yr/10),
    #     )

    figures = {}
    for (name, leg), profile in crew_profiles.items():
        fig = px.line(
            profile,
            x=x,
            y=y,
            color='Position',
            line_dash='Side',
            title=f"{name}, leg={leg}",
            template="plotly_white",
        )
        # fig.update_yaxes(
        #     range=yrange
        # )
        fig.update_layout(
            height=height,
            template=pio.templates.default,
        )
        figures[f"{name}, leg={leg}: {x}-{y}"] = fig
        # st.plotly_chart(fig, use_container_width=True)

    return figures, {}


def plot_crew_profile(piece_information, default_height=600, key='', input_container=None, cols=None):
    crew_profile = piece_information['crew_profile']
    n_legs = piece_information['n_legs']
    start_landmark = piece_information['start_landmark']
    finish_landmark = piece_information['finish_landmark']

    if not cols:
        input_container = input_container or st.container()
        with input_container:
            cols = st.columns(3)

    with cols[0]:
        x = st.selectbox(
            "Select x-axis",
            ['GateAngle', 'Normalized Time', 'GateForceX',
                'GateAngleVel', "GateAngle0"],
            key=key + "crew Select x-axis2",
        )
    with cols[1]:
        y = st.selectbox(
            "Select y-axis",
            ['GateForceX', 'GateAngle', 'GateAngleVel',
                "GateAngle0", 'Normalized Time'],
            key=key + "crew Select y-axis2",
        )
    with cols[2]:
        height = st.number_input(
            "Set figure height",
            min_value=100,
            max_value=None,
            value=default_height,
            step=100,
            key=key + "crew figure height"
        )

    fig = go.Figure()
    for (file, leg), piece_profile in crew_profile.groupby(["name", "leg"]):
        for (pos, side), profile in piece_profile.groupby(["Position", 'Side']):
            name = file if n_legs[file] == 1 else f"{file} {leg=}"
            fig.add_trace(
                go.Scatter(
                    x=profile[x],
                    y=profile[y],
                    legendgroup=f"{name} {leg} {side}",
                    legendgrouptitle_text=name,
                    name=f"{pos} {side}",
                    mode='lines',
                )
            )

    fig.update_layout(
        title=f"{start_landmark} to {finish_landmark}: {y} vs {x}",
        height=height,
        template=pio.templates.default,
        legend=dict(
            itemclick='toggle',
            itemdoubleclick='toggleothers',
            groupclick='toggleitem',
        )
    )
    # st.plotly_chart(fig, use_container_width=True)

    return {f'{x}-{y}': fig}, {}


def plot_boat_profile(piece_information, default_height=600, key="boat_", input_container=None, cols=None):
    boat_profiles = piece_information['boat_profiles']

    if not cols:
        with input_container or st.container():
            cols = st.columns(3)

    with cols[0]:
        facets = st.multiselect(
            "Select facets",
            [
                'Speed', 'Accel', 'Roll Angle', 'Pitch Angle', 'Yaw Angle'
            ],
            default=[
                'Speed', 'Accel', 'Roll Angle', 'Pitch Angle', 'Yaw Angle'
            ],
            key=key + "select_boat_facets"
        )
    with cols[1]:
        height = st.number_input(
            "Set figure height",
            min_value=100,
            max_value=None,
            value=len(facets) * 200,
            step=100,
            key=key + "select_boat_heigh"
        )

    boat_profile = pd.concat(
        boat_profiles, names=['name', 'leg']
    ).reset_index(["name", 'leg']).rename_axis(
        columns='Measurement'
    )
    boat_profile = boat_profile.loc[
        :, ~boat_profile.columns.duplicated()
    ].set_index(
        ["Normalized Time", "name", 'leg']
    )[facets].stack().rename("value").reset_index()
    fig = px.line(
        boat_profile,
        x="Normalized Time",
        y="value",
        color='name',
        line_dash='leg',
        facet_row='Measurement',
        template="plotly_white",
        # title=name
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_layout(height=height, template=pio.templates.default)

    return {'Boat profile': fig}, {}


def plot_piece_col(col, piece_information, default_height=600, key='piece', input_container=None):
    start_landmark = piece_information['start_landmark']
    finish_landmark = piece_information['finish_landmark']
    compare_power = piece_information['compare_power']
    landmark_distances = piece_information['landmark_distances']
    show_rowers = piece_information['show_rowers']
    piece_data_filter = piece_information['piece_data_filter']

    figures = {}
    tables = {}
    facet_col_wrap = 4

    if col == "Pace Boat":
        fig, time_behind = plot_pace_boat(
            piece_information['piece_data'],
            piece_information['landmark_distances'],
            piece_information['gps_data'],
            height=default_height,
            key=key,
            input_container=input_container,
        )
        figures[col] = fig
        tables["Time behind pace boat"] = time_behind
        # st.plotly_chart(fig, use_container_width=True)
        # st.subheader("Time behind pace boat")
        # st.dataframe(time_behind)
    else:
        if col == 'Work PC':
            with input_container or st.container():
                cols2 = st.columns(2)

            n_plots = len(
                compare_power[['name', 'leg', 'Position']].value_counts())
            if show_rowers:
                n_plots = len(show_rowers)

            with cols2[0]:
                facet_col_wrap = st.number_input(
                    "Select number of columns",
                    value=facet_col_wrap, min_value=1, step=1,
                )
                n_rows = np.ceil(n_plots / facet_col_wrap)

            with cols2[1]:
                default_height = st.number_input(
                    "Set Work PC figure height",
                    min_value=100,
                    # 3000,
                    value=int(default_height * n_rows // 2),
                    step=50,
                )

        fig = make_telemetry_distance_figure(
            compare_power, landmark_distances, col,
            facet_col_wrap=facet_col_wrap
        )
        itemclick = 'toggle'
        itemdoubleclick = "toggleothers"
        groupclick = 'toggleitem'
        fig.update_layout(
            title=f"{col}: {start_landmark} to {finish_landmark}",
            height=default_height,
            legend=dict(
                itemclick=itemclick,
                itemdoubleclick=itemdoubleclick,
                groupclick=groupclick,
            )
        )

        figures[col] = fig
        interval_stats = piece_data_filter.get(f"Average {col}")
        if interval_stats is not None:
            tables[f"Interval {col} Average"] = interval_stats

        average_stats = piece_data_filter.get(f"Average {col}")
        if average_stats is not None:
            tables[f"Piece {col} Average"] = average_stats
        # tables["Time behind pace boat"] = time_behind
        # st.plotly_chart(fig, use_container_width=True)

    return figures, tables
    # return {(col, f"{start_landmark} to {finish_landmark}"): fig}


@ st.cache_data
def interpolate_power(telemetry_data, dists=0.005, n_iter=10):
    power_gps_data = {}
    for k, data in telemetry_data.items():
        gps = data['positions']
        power = data['power']

        power_gps = gps.set_index('time')[
            ['longitude', 'latitude']
        ].apply(
            utils.interpolate_series, index=power.Time
        )
        power_gps = power.join(
            pd.concat({("boat", ""): power_gps},
                      axis=1).swaplevel(0, -1, axis=1)
        ).sort_index(axis=1)
        power_gps_data[k] = geodesy.interp_dataframe(
            power_gps, dists, n_iter=n_iter)

    return power_gps_data


@st.cache_data
def make_gps_heatmap(telemetry_data, dists, file_col, marker_size=5, map_style='open-street-map', height=600):

    power_gps_data = interpolate_power(telemetry_data, dists)

    fig = go.Figure()
    for k, data in power_gps_data.items():
        c = file_col[k]
        fig.add_trace(
            go.Scattermapbox(
                lon=data.longitude.squeeze(),
                lat=data.latitude.squeeze(),
                mode='lines+markers',
                marker=dict(
                    color=data[c].squeeze(),
                    coloraxis="coloraxis",
                    size=marker_size
                ),
                showlegend=True,
                name="|".join((k,) + c),
            )
        )

    fig.update_layout(
        mapbox={
            'style': map_style,
            'center': {
                'lon': data.longitude.squeeze().mean(),
                'lat': data.latitude.squeeze().mean(),
            },
            'zoom': 12
        },
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        showlegend=True,
        height=height,
    )

    return fig


def set_gps_heatmap(
        telemetry_data,
        datatype_container=None, data_container=None, settings_container=None,
        key='', default_height=1000,
):
    # cols = st.columns((4, 4, 2))
    datatype_container = datatype_container or st.container()
    data_container = data_container or st.container()
    settings_container = settings_container or st.container()
    with settings_container:
        dists = st.number_input(
            "marker spacing (m)", min_value=1, value=5,
            key='heatmap_spacing' + key
        ) / 1000
        height = st.number_input(
            "Heatmap height (px)", min_value=100, step=50, value=default_height,
            key='heatmap_height' + key
        )
        marker_size = st.number_input(
            "Marker size", min_value=1, value=10, key='heatmap_size' + key
        )
        map_style = st.selectbox(
            "map style",
            ["open-street-map", "carto-positron", "carto-darkmatter"],
            key='heatmap_style' + key
        )
        colorscales = px.colors.named_colorscales()
        colorscale = st.selectbox(
            "heatmap color scale",
            colorscales,
            index=colorscales.index('plasma'),
            key='heatmap_colorscale' + key
        )
        cmin = st.number_input(
            "Color scale min (clear for autoscaling)", value=None, key='heatmap_cmin' + key)
        cmid = st.number_input(
            "Color scale mid (clear for autoscaling)", value=None, key='heatmap_cmid' + key)
        cmax = st.number_input(
            "Color scale max (clear for autoscaling)", value=None, key='heatmap_cmax' + key)
        zoom = st.number_input(
            "Zoom level",
            min_value=0., max_value=30., value=12., step=1.,
            key='heatmap_zoom' + key
        )

    c0 = 'AvgBoatSpeed'
    c1 = 'Boat'
    file_col = {}
    for k, data in telemetry_data.items():
        data = data['power']
        with datatype_container:
            _options = [
                'Angle 0.7 F', 'Angle Max F', 'Average Power', 'AvgBoatSpeed',
                'CatchSlip', 'Dist/Stroke', 'Drive Start T', 'Drive Time',
                'Effective', 'FinishSlip', 'Length', 'Max Force PC', 'MaxAngle',
                'MinAngle', 'Rating', 'Recovery Time', 'Rower Swivel Power',
                'StrokeNumber', 'SwivelPower''Work PC Q1', 'Work PC Q2',
                'Work PC Q3', 'Work PC Q4'
            ]
            options = data.columns.levels[0].intersection(_options)
            index = (
                int(options.get_indexer_for([c0])[0]) if c0 in options else 0)
            c0 = st.selectbox(
                f"choose data type to plot for {k}",
                options=options,
                index=index,
                key=f'heatmap_datatype_{k}' + key
            )

        with data_container:
            options = data[c0].columns.get_level_values(0)
            index = (
                int(options.get_indexer_for([c1])[0]) if c1 in options else 0)
            c1 = st.selectbox(
                f"choose data to plot for {k}",
                options=options,
                index=index,
                key=f'heatmap_data_{k}' + key
            )

        file_col[k] = (c0, c1)

    fig = make_gps_heatmap(
        telemetry_data, dists, file_col, marker_size=marker_size, map_style=map_style, height=height)
    fig.update_coloraxes(
        cmin=cmin,
        cmid=cmid,
        cmax=cmax,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            outlinewidth=0,
        )
    )
    fig.update_layout(mapbox={'zoom': zoom})

    return fig, file_col


def make_static_report(report_outputs, file_name):
    static_report = static.StreamlitStaticExport()
    for (i, header), outputs in report_outputs.items():
        static_report.add_header(i, header, 'H2')
        for keys, output in outputs.items():
            key = "-".join(keys[1:])
            static_report.add_header(f"{i}-{key}", keys[-1], 'H3')
            if keys[1] == 'figure':
                output.update_layout(template='plotly_white')
                static_report.export_plotly_graph(
                    key, output, include_plotlyjs='cdn')
            elif keys[1] == 'table':
                static_report.export_dataframe(key, output)

    st.download_button(
        f":inbox_tray: {file_name}",
        static_report.create_html(),
        file_name,
        mime='text/html',
    )
