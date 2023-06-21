
import streamlit as st

import logging
import datetime
import time 

import sys 
import os
from pathlib import Path 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm.autonotebook import tqdm

import plotly as pl
from plotly import express as px, subplots

DIRPATH = Path(__file__).resolve().parent
LIBPATH = str(DIRPATH.parent.parent)
realpaths = [os.path.realpath(p) for p in sys.path]
if LIBPATH not in realpaths:
    sys.path.append(LIBPATH)
    print("adding", LIBPATH)

from rowing.world_rowing import api, live, utils, fields
from rowing.app import select, inputs, state, plots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

st.set_page_config(
    page_title="World Rowing Realtime Livetracker",
    layout='wide'
    # page_icon="👋",
)


def main(params=None):
    state.update(params or {})
    st.title("World Rowing Realtime Livetracker")

    state.reset_button()

    with st.sidebar:
        with st.expander("Settings"):
            replay = st.checkbox(
                "replay race data", state.get("replay", False))
            replay_step = st.number_input(
                "replay step", 1, 100, state.get("replay_step", 10))
            realtime_sleep = st.number_input(
                "poll", 0., 10., state.get("poll", 3.), step=0.5
            )
            fig_height = st.number_input(
                "plot size", 10, 2_000, 1000
            )
            fig_autosize = st.checkbox("autosize plot")

            clear = st.button("clear cache")
            if clear:
                st.cache_data.clear()

    if replay:
        races = select.select_races(
            filters=True, select_all=True
        )
        race = inputs.select_dataframe(races, fields.Race)
    else:
        races = api.get_live_races()
        if races.empty:
            next_races = api.get_next_races(5)
            if not next_races.empty:
                st.write("next races:")
                st.dataframe(fields.to_streamlit_dataframe(next_races))

                cols = st.columns(2)
                if cols[0].checkbox("refresh until next race"):
                    with cols[1]:
                        countdown = st.empty()
                        for t in range(10, -1, -1):
                            countdown.metric(
                                "Refresh in", f"{t} s"
                            )
                            time.sleep(1)
                        st.experimental_rerun()
                if st.button("refresh"):
                    st.experimental_rerun()

            st.write(
                "no live race could be loaded, "
                "check replay in sidebar to see race replay")
            st.stop()

        race = inputs.select_dataframe(races, fields.Race)
        # race = api.get_live_race()
        if race is None:
            st.write("no live race could be loaded")
            if st.checkbox("refresh until next"):
                time.sleep(10)
                st.experimental_rerun()

            if st.button("refresh"):
                st.experimental_rerun()
            st.stop()

    st.write(race)

    race_id = race.race_id

    get_live_race_data = st.cache_resource(live.LiveRaceData)

    live_race = get_live_race_data(
        race_id,
        realtime_sleep=realtime_sleep,
        replay=replay,
        replay_step=replay_step
    )
    show_intermediates = st.empty()
    fig_plot = st.empty()

    facets = [
        fields.live_raceBoatTracker_distanceFromLeader,
        fields.split,
        fields.live_raceBoatTracker_metrePerSecond,
        fields.live_raceBoatTracker_strokeRate,
    ]
    with utils.ThreadPoolExecutor(max_workers=5) as executor, tqdm() as pbar:
        updates = utils.WorkQueue(executor, queue_size=1).run(
            live_race.update, live_race.tracker.run()
        )
        for i, (live_race, data) in enumerate(updates):
            pbar.update(1)
            plot_data = live_race.plot_data(facets)
            intermediates = live_race.intermediates

            with show_intermediates:
                cols = st.columns(2)
                with cols[0]:
                    if live_race.lane_info is not None:
                        plots.show_lane_info(live_race.lane_info)
                        # speed_col = fields.lane_currentPoint_raceBoatTracker_metrePerSecond
                        # dist_col = fields.lane_currentPoint_raceBoatTracker_distanceTravelled
                        # pos_col = fields.lane_currentPoint_raceBoatTracker_currentPosition
                        # rate_col = fields.lane_currentPoint_raceBoatTracker_strokeRate
                        # lane_info = live_race.lane_info[[
                        #     fields.lane_Lane, 
                        #     # pos_col,
                        #     dist_col, 
                        #     fields.lane_currentPoint_raceBoatTracker_strokeRate, 
                        #     speed_col, 
                        # ]].sort_values(fields.lane_Lane).copy()
                        # lane_info[fields.split] = pd.to_timedelta(
                        #     500 / lane_info[speed_col], unit='s',
                        # ) + pd.Timestamp(0)
                        # # lane_info[dist_col] = lane_info[dist_col].astype(float)
                        # st.dataframe(
                        #     lane_info,
                        #     column_config = {
                        #         pos_col: st.column_config.ProgressColumn(
                        #             pos_col, 
                        #             min_value=0, 
                        #             max_value=len(lane_info),
                        #             format="%d"
                        #         ), 
                        #         speed_col: st.column_config.ProgressColumn(
                        #             speed_col, 
                        #             help='Metre Per Second',
                        #             min_value=float(live_race.lane_info[speed_col].min()) - 0.1, 
                        #             max_value=float(live_race.lane_info[speed_col].max()),
                        #             format="%.1f"
                        #         ), 
                        #         dist_col: st.column_config.ProgressColumn(
                        #             dist_col, 
                        #             min_value=int(live_race.lane_info[dist_col].min()) - 1, 
                        #             max_value=int(live_race.lane_info[dist_col].max()),
                        #             format="%.0d"
                        #         ), 
                        #         rate_col: st.column_config.ProgressColumn(
                        #             rate_col, 
                        #             min_value=int(live_race.lane_info[rate_col].min()) - 1, 
                        #             max_value=int(live_race.lane_info[rate_col].max()),
                        #             format="%.0d"
                        #         ), 
                        #         fields.split: st.column_config.TimeColumn(
                        #             fields.split, 
                        #             # min_value=(lane_info[fields.split].min()), 
                        #             # max_value=(lane_info[fields.split].max()),
                        #             format="m:ss"
                        #         ), 
                        #     }, 
                        #     use_container_width=True
                        # )

                if intermediates is not None and not intermediates.empty:
                    with cols[1]:
                        inter_pos = intermediates[fields.intermediates_Rank]
                        row = pd.DataFrame(
                            [inter_pos.columns], 
                            index=[fields.intermediates_ResultTime],
                            columns=inter_pos.columns, 
                        )
                        inter_time = fields.to_streamlit_dataframe(
                            intermediates[fields.intermediates_ResultTime]
                        )
                        inters = pd.concat([inter_pos, row, inter_time], axis=0)     
                        inters.index.name = fields.intermediates_Rank
                        st.dataframe(
                            inters,
                            use_container_width=True
                        )
                        # st.dataframe(, use_container_width=True)

           
            if plot_data is not None:
                facet_rows = {facet: len(facets) - i for i, facet in enumerate(facets)}
                plot_data, facet_axes, facet_format = plot_data
                fig = px.line(
                    plot_data,
                    x=fields.live_raceBoatTracker_distanceTravelled,
                    y='value',
                    color=fields.raceBoats,
                    facet_row="live",
                    category_orders={
                        "live": facets,
                    },
                    # color_discrete_map=dict(zip(
                    #     live_race.lane_info[fields.lane_Lane].sort_values().index, 
                    #     px.colors.DEFAULT_PLOTLY_COLORS
                    # )),
                    hover_data=facet_format,
                )
                for facet, row in facet_rows.items():
                    fig.update_yaxes(row=row, **facet_axes[facet])

                fig.update_annotations(text="")
                if fig_autosize:
                    fig.update_layout(autosize=True)
                else:
                    fig.update_layout(height=fig_height)

                with fig_plot:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                with fig_plot:
                    st.write("no live data could be loaded")

    state.update_query_params()
    return state.get_state()


if __name__ == "__main__":
    main()
