
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.validators.scatter.marker import SymbolValidator

from rowing.world_rowing import fields

FORMATS = {
    "default": {
        "format": ':.1%'
    },
    "percentage": {
        "format": ':.1%',
        "tickformat": ',.1%',
    },
    "time": {
        "format": "|%-M:%S.%L",
        "tickformat": "%-M:%S",
    }
}

SYMBOLS = [
    s for s in SymbolValidator().values[2::3]
    if not s.endswith("open") or s.endswith("dot")
]


def live_race_plot_data(live_race, *args, **kwargs):
    if live_race.livetracker is None:
        return None

    return melt_livetracker(
        live_race.livetracker,
        live_race.lanes.index,
        live_race.race_distance,
        **kwargs
    )


def melt_livetracker(
        livetracker, lanes=None, race_distance=2000, filter_distance=100
):
    facets = [
        fields.live_raceBoatTracker_distanceFromLeader,
        fields.split,
        # fields.live_raceBoatTracker_metrePerSecond,
        fields.live_raceBoatTracker_strokeRate,
    ]
    facet_rows = {facet: len(facets) - i for i, facet in enumerate(facets)}
    index_names = [
        fields.live_raceBoatTracker_id,
        fields.raceBoats,
        fields.live_raceBoatTracker_distanceTravelled
    ]
    lanes = livetracker.columns.levels[1] if lanes is None else lanes

    stacked = livetracker.stack(
        1
    ).reindex(
        pd.MultiIndex.from_product([livetracker.index, lanes])
    ).droplevel(0).reset_index().dropna(
        subset=index_names
    ).set_index(
        index_names
    )

    plot_data = stacked[
        facets
    ].reset_index().melt(
        index_names, var_name='facet',
    ).join(stacked[facets], on=index_names)

    plot_data = fields.to_plotly_dataframe(plot_data.dropna(subset=["value"]))

    facet_format, facet_axes, facet_data = facet_properties(
        plot_data, race_distance=race_distance, filter_distance=filter_distance,
        format={fields.split: "|%-M:%S.%L"},
    )

    return plot_data, facet_rows, facet_axes, facet_format


def select_figure_params():
    fig_height = st.number_input(
        "plot size", 10, 2_000, 1000
    )
    fig_autosize = st.checkbox("autosize plot")

    params = {}
    params['layout'] = layout = {}

    layout['legend'] = dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
    if fig_autosize:
        layout["autosize"] = True
    else:
        layout["height"] = fig_height

    return params


def update_figure(
    fig, layout=None, xaxes=None, yaxes=None, annotations=None,
):
    if fig is None:
        return None

    if layout:
        fig.update_layout(**layout)
    if xaxes:
        fig.update_xaxes(**xaxes)
    if yaxes:
        fig.update_yaxes(**yaxes)
    if annotations:
        fig.update_annotations(**annotations)

    return fig


def make_plots(
    fig_data, *args, **kwargs
):
    if fig_data is None:
        return None

    plot_data, facet_rows, facet_axes, facet_format = fig_data
    fig = px.line(
        plot_data,
        x=fields.live_raceBoatTracker_distanceTravelled,
        y='value',
        color=fields.raceBoats,
        facet_row="facet",
        category_orders={
            "facet": list(facet_rows),
        },
        hover_data=facet_format,
    )
    for facet, row in facet_rows.items():
        fig.update_yaxes(row=row, **facet_axes[facet])

    fig.update_annotations(text="")
    return update_figure(fig, **kwargs)


def melt_livetracker_times(live_data, filter_distance=100):
    live_data = fields.to_plotly_dataframe(live_data)
    plot_data = live_data.dropna(
        subset=[
            fields.live_raceBoatTracker_distanceTravelled,
            fields.split,
            fields.avg_split,
        ]
    ).melt([
        fields.live_time,
        fields.live_raceBoatTracker_distanceTravelled,
        fields.raceBoats,
        fields.boatClass,
        fields.live_raceId,
        fields.race_distance,
    ], var_name='facet')
    plot_data[fields.crew] = (
        plot_data[fields.raceBoats] + " " + plot_data[fields.boatClass])
    plot_data = plot_data.merge(
        live_data, on=[
            fields.live_time,
            fields.live_raceBoatTracker_distanceTravelled,
            fields.raceBoats,
            fields.live_raceId,
        ],
        suffixes=("", "_1")
    )

    facet_format, facet_axes, facet_data = facet_properties(
        plot_data,
        race_distance=plot_data[fields.race_distance],
        filter_distance=filter_distance,
        format={
            fields.avg_speed: ":0.1f",
            fields.distance_from_pace: ":0.1f",
            fields.PGMT: ':.1%',
            fields.split: "|%-M:%S.%L",
            fields.avg_split: "|%-M:%S.%L",
            fields.lane_ResultTime: "|%-M:%S.%L",
        },
    )

    return plot_data, facet_axes, facet_format


def facet_properties(
    plot_data, filter_distance=100, race_distance=2000, quantile=0.2,
    format=None, axes=None
):

    distance_travelled = plot_data[fields.live_raceBoatTracker_distanceTravelled]
    filter_distance = min(
        distance_travelled.quantile(quantile), filter_distance)
    filter = (
        (distance_travelled > filter_distance)
        & (distance_travelled < race_distance)
        & plot_data.facet.notnull()
    )
    if not filter.any():
        facet_groups = plot_data.groupby("facet")
    else:
        facet_groups = plot_data[filter].groupby("facet")

    facet_types = facet_groups.value.first().map(type)
    facets = facet_types.index[facet_types != str]

    facet_max = facet_groups.value.max()[facets]
    facet_min = facet_groups.value.min()[facets]
    facet_ptp = facet_max - facet_min
    facet_data = pd.concat(
        [facet_min - facet_ptp*0.1, facet_max + facet_ptp*0.1, ],
        axis=1).apply(tuple, axis=1).rename("range").to_frame()
    facet_data['matches'] = None
    facet_data['title_text'] = facet_data.index

    facet_format = {f: True for f in facets}

    if format:
        facet_format.update(format)

    facet_axes = facet_data.T.to_dict()
    for facet in facet_axes:
        fmt = facet_format[facet]
        if fmt is not True:
            facet_axes[facet]['tickformat'] = fmt

    facet_axes.setdefault(fields.PGMT, {})['tickformat'] = ',.1%'
    facet_axes.setdefault(fields.split, {})['tickformat'] = "%-M:%S"
    facet_axes.setdefault(fields.avg_split, {})['tickformat'] = "%-M:%S"
    facet_axes.setdefault(fields.lane_ResultTime, {})['tickformat'] = "%-M:%S"

    for col in [
        fields.distance_from_pace,
        fields.live_raceBoatTracker_distanceFromLeader,
        fields.split,
        fields.avg_split,
    ]:
        if col in facet_axes and 'range' in facet_axes[col]:
            facet_axes[col]['range'] = facet_axes[col]['range'][::-1]

    if axes:
        facet_axes.update(axes)

    return facet_format, facet_axes, facet_data


def make_livetracker_plot(
    facets, plot_data, facet_axes, facet_format,
    facet_row_spacing=0.01, height=1000, width=800, **kwargs
):
    facet_rows = {facet: len(facets) - i for i, facet in enumerate(facets)}
    fig = px.line(
        plot_data[plot_data.facet.isin(facets)],
        x=fields.live_raceBoatTracker_distanceTravelled,
        y="value",
        color=fields.crew,
        hover_data=facet_format,
        facet_row="facet",
        category_orders={
            "facet": facets,
        },
        facet_row_spacing=facet_row_spacing,
        height=height,
        width=width,
        **kwargs
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    fig.update_annotations(text="")
    for facet, row in facet_rows.items():
        fig.update_yaxes(row=row, **facet_axes[facet])

    return fig


def show_intermediates(intermediates):
    inter_pos = intermediates[fields.intermediates_Rank]
    row = pd.DataFrame(
        [inter_pos.columns],
        index=[fields.Time],
        columns=inter_pos.columns,
    )
    inter_time = fields.to_streamlit_dataframe(
        intermediates[fields.intermediates_ResultTime]
    )
    inters = pd.concat([inter_pos, row, inter_time], axis=0)
    inters.index.name = 'Rank'
    st.dataframe(
        inters, use_container_width=True
    )


def show_lane_info(lane_info):
    speed_col = fields.lane_currentPoint_raceBoatTracker_metrePerSecond
    dist_col = fields.lane_currentPoint_raceBoatTracker_distanceTravelled
    pos_col = fields.lane_currentPoint_raceBoatTracker_currentPosition
    rate_col = fields.lane_currentPoint_raceBoatTracker_strokeRate
    lane_info = lane_info[[
        fields.lane_Lane,
        # pos_col,
        dist_col,
        fields.lane_currentPoint_raceBoatTracker_strokeRate,
        speed_col,
    ]].sort_values(fields.lane_Lane).copy()
    lane_info[fields.split] = pd.to_timedelta(
        500 / lane_info[speed_col], unit='s',
    ) + pd.Timestamp(0)
    st.dataframe(
        lane_info,
        column_config={
            pos_col: st.column_config.ProgressColumn(
                pos_col,
                min_value=0,
                max_value=len(lane_info),
                format="%d"
            ),
            speed_col: st.column_config.ProgressColumn(
                speed_col,
                help='Metre Per Second',
                min_value=float(lane_info[speed_col].min()) - 0.1,
                max_value=float(lane_info[speed_col].max()),
                format="%.1f"
            ),
            dist_col: st.column_config.ProgressColumn(
                dist_col,
                min_value=int(lane_info[dist_col].min()) - 1,
                max_value=int(lane_info[dist_col].max()),
                format="%d"
            ),
            rate_col: st.column_config.ProgressColumn(
                rate_col,
                min_value=int(lane_info[rate_col].min()) - 1,
                max_value=int(lane_info[rate_col].max()),
                format="%d"
            ),
            fields.split: st.column_config.TimeColumn(
                fields.split,
                format="m:ss"
            ),
        },
        use_container_width=True
    )


def show_lane_intermediates(lane_info, intermediates):
    cols = st.columns(2)
    with cols[0]:
        if lane_info is not None and not lane_info.empty:
            show_lane_info(lane_info)

    with cols[1]:
        if intermediates is not None and not intermediates.empty:
            show_intermediates(intermediates)
