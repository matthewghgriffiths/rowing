import streamlit as st
import io
import zipfile
import logging
from itertools import count

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px

from rowing.analysis import splits, files, geodesy, telemetry
from rowing import utils
from rowing.app import threads


logger = logging.getLogger(__name__)

color_discrete_sequence = [
    "#0068c9",
    "#83c9ff",
    "#ff2b2b",
    "#ffabab",
    "#29b09d",
    "#7defa1",
    "#ff8700",
    "#ffd16a",
    "#6d3fc0",
    "#d5dae5",
]

DEFAULT_REPORT = {
    "report_0": {
        "pace boat time input": None,
        "select piece": "Pace Boat",
        "select plot": "Piece profile"
    },
    "report_1": {
        "select piece": "AvgBoatSpeed",
        "select plot": "Piece profile"
    },
    "report_2": {
        "select piece": "Rating",
        "select plot": "Piece profile"
    },
    "report_3": {
        "select piece": "Rower Swivel Power",
        "select plot": "Piece profile"
    },
    "report_4": {
        "select piece": "MinAngle",
        "select plot": "Piece profile"
    },
    "report_5": {
        "select piece": "MaxAngle",
        "select plot": "Piece profile"
    },
    "report_6": {
        "select piece": "CatchSlip",
        "select plot": "Piece profile"
    },
    "report_7": {
        "select piece": "FinishSlip",
        "select plot": "Piece profile"
    },
    "report_8": {
        "select piece": "Effective",
        "select plot": "Piece profile"
    },
    "report_9": {
        "Select x-axis": "GateAngle",
        "Select y-axis": "GateForceX",
        "rower profile figure height": 600,
        "select plot": "Stroke profile",
        "select stroke": "Rower profile"
    },
    "report_10": {
        "Select x-axis": "GateAngle",
        "Select y-axis": "GateAngleVel",
        "rower profile figure height": 600,
        "select plot": "Stroke profile",
        "select stroke": "Rower profile"
    },
    "report_11": {
        "Select x-axis": "Normalized Time",
        "Select y-axis": "GateForceX",
        "rower profile figure height": 600,
        "select plot": "Stroke profile",
        "select stroke": "Rower profile"
    },
    "report_12": {
        "select plot": "Stroke profile",
        "select stroke": "Boat profile",
        "select_boat_facets": [
            "Speed",
            "Accel",
            "Roll Angle",
            "Pitch Angle",
            "Yaw Angle"
        ],
        "select_boat_heigh": 1000
    },
    "report_setup": {
        "figure_height": 1000,
        "nview": 13,
        "toggleother": False,
        "window": 10
    }
}


@st.cache_data
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


@st.cache_data
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
def parse_excel(file, use_names=True):
    data = pd.read_excel(file, header=None)
    return telemetry.parse_powerline_excel(data, use_names=use_names)


@st.cache_data
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


@st.cache_data
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


@st.cache_data
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


@st.cache_data
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


@st.cache_data
def get_fastest_times(gpx_data):
    best_times, errors = utils.map_concurrent(
        splits.find_all_best_times,
        gpx_data, singleton=True,
    )
    if errors:
        logging.error(errors)
    return best_times


def select_pieces(all_crossing_times):
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

    landmark_distance = sel_times.reset_index(
        'distance'
    ).distance
    landmarks = landmark_distance.groupby(
        level=2).mean().sort_values().index
    landmark_dist = landmark_distance.unstack(
        'landmark'
    ).dropna(axis=1).mean(0)

    if landmark_dist.size < 2:
        landmark_distance = sel_times.sort_values().reset_index("distance").distance
        landmark_dist = landmark_distance.unstack(
            'landmark').mean(0)
        landmarks = landmark_distance.index.get_level_values(
            "landmark").drop_duplicates()

    start, end = map(
        int, landmarks.get_indexer(
            [landmark_dist.idxmin(), landmark_dist.idxmax()]
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


def show_piece_data(piece_data):
    tabs = st.tabs(list(piece_data))
    for tab, (key, data) in zip(tabs, piece_data.items()):
        with tab:
            data = data.copy()
            for c, col in data.items():
                if pd.api.types.is_timedelta64_dtype(col.dtype):
                    data[c] = col.map(utils.format_timedelta)

            st.dataframe(data)


def align_pieces(piece_data, start_landmark, finish_landmark, gps_data, resolution=0.005):
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


def set_landmarks(gps_data=None, landmarks=None, title=True):
    tab0, tab1, tab2 = st.tabs([
        "From Pieces",
        "Edit Landmarks",
        "Upload Landmarks",  # "Map of Landmarks"
    ])
    with tab0:
        new_landmarks = {}
        if not gps_data:
            st.write("no gps data loaded")
        else:
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
                        dist = st.slider(
                            "Select distance",
                            min_value=data.distance.min(),
                            max_value=data.distance.max(),
                            step=0.001,
                            format="%.3f km",
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

    # new_locations = pd.DataFrame()
    if new_landmarks:
        new_locations = pd.concat(
            new_landmarks, names=['location', 'landmark'], axis=0
        ).reset_index("distance", drop=True).reset_index()
        landmarks = pd.concat(
            [new_locations, landmarks]
        )
        download_csv(
            "piece_landmarks.csv",
            new_locations,
            ':inbox_tray: download piece landmarks as csv',
            csv_kws=dict(index=False),
        )

    with tab2:
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

    # with tab3:
    if True:
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
        fig = go.Figure()

        if gps_data:
            for name, data in gps_data.items():
                fig.add_trace(go.Scattermapbox(
                    lon=data.longitude,
                    lat=data.latitude,
                    mode='lines',
                    name=name,
                ))

        fig.add_trace(go.Scattermapbox(
            lon=set_landmarks.longitude,
            lat=set_landmarks.latitude,
            # customdata = set_landmarks,
            mode='markers+text',
            name='Landmarks',
            text=set_landmarks.landmark,
            cluster=dict(
                enabled=True,
                maxzoom=5,
                step=1,
                size=20,
            ),
            marker={
                'size': 5,
                'color': 'black',
            },
            legendgroup='Landmarks',
            legendgrouptitle_text='Landmarks',
            textposition='bottom right',
        ))

        for i, landmark in set_landmarks.iterrows():
            arrow = geodesy.make_arrow_base(landmark, 0.25, 0.1, 20)
            color = 'black'
            if (landmark.location, landmark.landmark) in new_landmarks:
                color = 'red'

            trace = go.Scattermapbox(
                lon=arrow.longitude,
                lat=arrow.latitude,
                mode='lines',
                name=landmark.landmark,
                fill='toself',
                hovertext=f"bearing={landmark.bearing:.1f}",
                line=dict(
                    width=3,
                ),
                marker={"color": color},
                textposition='bottom right',
                showlegend=False,
                legendgroup='Landmarks',
                legendgrouptitle_text='Landmarks',
            )
            fig.add_trace(trace)

        lon = set_landmarks.longitude.mean()
        lat = set_landmarks.latitude.mean()
        zoom = 5

        if gps_data:
            positions = pd.concat(gps_data)
            lon = positions.longitude.mean()
            lat = positions.latitude.mean()
            zoom = 15 - max(
                np.ptp(positions.longitude),
                np.ptp(positions.latitude),
            ) * 111

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
                x=0.01
            ),
            height=height,
            overwrite=True
        )
        st.plotly_chart(fig, use_container_width=True)

    return set_landmarks


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
        profile = telemetry_data[name]['Periodic']
        start_time = piece_times.min()
        finish_time = piece_times.max()
        piece_profile = profile[
            profile.Time.dt.tz_localize(None).between(start_time, finish_time)
        ].set_index('Time').dropna(axis=1, how='all')

        profiles[name, leg] = profile = telemetry.norm_stroke_profile(
            piece_profile, nres)

        gate_angle = profile.GateAngle
        gate_angle0 = gate_angle - gate_angle.values.mean(0, keepdims=True)
        for pos, angle0 in gate_angle0.items():
            profile["GateAngle0", pos] = angle0

        mean_profile = profile.groupby(
            level=1
        ).mean().reset_index().rename(
            {"": "Boat"}, axis=1, level=1
        )
        boat_profiles[name, leg] = boat_profile = mean_profile.xs(
            "Boat", axis=1, level=1)

        profile = mean_profile[
            mean_profile.columns.levels[0].difference(
                boat_profile.columns)
        ].rename_axis(
            columns=("Measurement", "Position")
        ).stack(1).reset_index(
            "Position"
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


@st.cache_data
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
            template="streamlit",
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


@st.cache_data
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
            # title=name,
            # color_discrete_sequence=app.color_discrete_sequence,
        )
    else:
        fig = go.Figure()
        for (file, leg), data in compare_power.groupby(["name", "leg"]):
            pos_power = data.dropna(
                subset=col).groupby("Position")
            for pos, pos_data in pos_power:
                name = file if n_legs[file] == 1 else f"{file} {leg=}"
                fig.add_trace(
                    go.Scatter(
                        x=pos_data["Distance"],
                        y=pos_data[col],
                        legendgroup=f"{file} {leg}",
                        legendgrouptitle_text=name,
                        name=pos,
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


@st.cache_data
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


@st.cache_data
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
                value=state.get('Make all plots'),
                key=key + 'Make all plots')
        elif nview:
            all_plots = st.number_input(
                "Number of panels",
                min_value=0,
                value=st.session_state.get(key + 'nview', 0),
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
            title=f"{name}, leg={leg}"
        )
        # fig.update_yaxes(
        #     range=yrange
        # )
        fig.update_layout(
            height=height
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
        for pos, profile in piece_profile.groupby("Position"):
            name = file if n_legs[file] == 1 else f"{file} {leg=}"
            fig.add_trace(
                go.Scatter(
                    x=profile[x],
                    y=profile[y],
                    legendgroup=f"{name} {leg}",
                    legendgrouptitle_text=name,
                    name=pos,
                    mode='lines',
                )
            )

    fig.update_layout(
        title=f"{start_landmark} to {finish_landmark}: {y} vs {x}",
        height=height,
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
    ).set_index(
        ["Normalized Time", "name", 'leg']
    )[facets].stack().rename("value").reset_index()
    fig = px.line(
        boat_profile,
        x="Normalized Time",
        y="value",
        color='name',
        line_dash='leg',
        facet_row='Measurement',
        # title=name
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_layout(height=height)

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


@st.cache_data
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
            pd.concat({"boat": power_gps}, axis=1).swaplevel(0, 1, axis=1)
        )
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
