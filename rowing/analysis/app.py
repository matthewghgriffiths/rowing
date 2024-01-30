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
        logging.error(errs)

    return data


@st.cache_data
def parse_excel(file, use_names=True):
    data = pd.read_excel(file, header=None)
    return telemetry.parse_powerline_excel(data, use_names=use_names)


@st.cache_data
def parse_telemetry_excel(uploaded_files, use_names=True):
    uploaded_data = {
        file.name.rsplit(".", 1)[0]: file.read().decode("utf-8")
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
    return crossing_times


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
    return piece_data, start_landmark, finish_landmark, intervals


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
            if 'npick_distance' not in st.session_state:
                st.session_state.setdefault("npick_distance", 0)
                # st.session_state.npick_distance = 0

            cols = st.columns((1, 5))
            with cols[0]:
                st.markdown("<br>", unsafe_allow_html=True)
                count = st.empty()
                if st.button("Add piece landmark"):
                    st.session_state.npick_distance += 1
                    # st.experimental_rerun()
                if st.button("Remove piece landmark"):
                    st.session_state.npick_distance -= 1
                    # st.experimental_rerun()
                with count:
                    npick = st.session_state.get("npick_distance", 0)
                    st.write(
                        f"\nSetting {npick} landmarks from pieces")

            with cols[1]:
                for i in range(st.session_state.get("npick_distance", 0)):
                    cols = st.columns(3)
                    with cols[0]:
                        name = st.selectbox(
                            "Pick piece",
                            options=list(gps_data.keys()),
                            key=f"Pick piece {i}",
                        )
                        data = gps_data[name]
                    with cols[1]:
                        dist = st.select_slider(
                            "Select distance",
                            # value=data.distance.sample().iloc[0],
                            options=data.distance,
                            format_func="{:.3f} km".format,
                            key=f"Pick piece distance {i}",
                        )
                    with cols[2]:
                        landmark = st.text_input(
                            "Enter landmark name",
                            value=f"{name} {dist:.3f} km",
                            key=f"Pick piece landmark {i}",
                        )

                    new_landmarks[name, landmark] = data.set_index("distance").loc[
                        [dist], ['latitude', 'longitude', 'bearing']]

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
                # 'symbol': "airfield",
                # 'icon': dict(iconUrl="https://api.iconify.design/maki-city-15.svg"),
            },
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
                # hoverinfo = landmark_locs.index,
                mode='lines',
                name=landmark.landmark,
                fill='toself',
                hovertext=f"bearing={landmark.bearing:.1f}",
                line=dict(
                    width=3,
                ),
                marker={"color": color},
                textposition='bottom right',
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
        # if new_landmarks:
        #     lon = new_locations.longitude.mean()
        #     lat = new_locations.latitude.mean()
        #     if len(new_locations) > 1:
        #         zoom = 15 - max(
        #             np.ptp(new_locations.longitude),
        #             np.ptp(new_locations.latitude),
        #         ) * 111

        fig.update_layout(
            {"uirevision": True},
            mapbox={
                'style': map_style,
                'center': {'lon': lon, 'lat': lat},
                'zoom': zoom
            },
            showlegend=False,
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
    data = locations
    for name, data in gps_data.items():
        fig.add_trace(go.Scattermapbox(
            lon=data.longitude,
            lat=data.latitude,
            mode='lines',
            name=name,
        ))
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
            orientation="h",
            yanchor="top",
            y=-.02,
            xanchor="right",
            x=1
        ),
        height=height,
    )
    return fig


@st.cache_data
def make_stroke_profiles(telemetry_data, piece_data, nres=101):
    profiles = {}
    boat_profiles = {}
    crew_profiles = {}
    for piece, piece_times in piece_data['Timestamp'].iterrows():
        name = piece[1]
        profile = telemetry_data[name]['Periodic']
        start_time = piece_times.min()
        finish_time = piece_times.max()
        piece_profile = profile[
            profile.Time.dt.tz_localize(None).between(start_time, finish_time)
        ].set_index('Time').dropna(axis=1, how='all')

        profiles[name] = profile = telemetry.norm_stroke_profile(
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
        boat_profiles[name] = boat_profile = mean_profile.xs(
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
        crew_profiles[name] = profile

    return profiles, boat_profiles, crew_profiles


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
