
import streamlit as st

import logging
import datetime

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
            dummy = st.checkbox(
                "load dummy data", state.get("dummy", False))
            dummy_index = st.number_input("dummy_index", 0, 1000, 0)
            dummy_step = st.number_input(
                "dummy_step", 1, 100, state.get("dummy_step", 10))
            realtime_sleep = st.number_input(
                "poll", 0, 5, 3
            )

    if dummy:
        races = select.select_races(
            filters=True, select_all=True
        )
        race = inputs.select_dataframe(races, fields.Race)
    else:
        races = api.get_live_races()
        if races.empty:
            st.write("no live race could be loaded")
            if st.button("refresh"):
                st.experimental_rerun()
            st.stop()

        race = inputs.select_dataframe(races, fields.Race)
        # race = api.get_live_race()
        if race is None:
            st.write("no live race could be loaded")
            if st.button("refresh"):
                st.experimental_rerun()
            st.stop()

    st.write(race)

    race_id = race.race_id

    get_live_race_data = st.cache_resource(live.LiveRaceData)

    live_race = get_live_race_data(
        race_id,
        realtime_sleep=realtime_sleep,
        dummy=dummy,
        dummy_index=dummy_index,
        dummy_step=dummy_step
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
            if intermediates is not None and not intermediates.empty:
                with show_intermediates:
                    cols = st.columns(2)
                    print(intermediates)
                    with cols[0]:
                        st.dataframe(
                            intermediates[fields.intermediates_Rank]
                        )
                    with cols[1]:
                        st.dataframe(fields.to_streamlit_dataframe(
                            intermediates[fields.intermediates_ResultTime]
                        ))


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
                    hover_data=facet_format,
                )
                for facet, row in facet_rows.items():
                    fig.update_yaxes(row=row, **facet_axes[facet])
                with fig_plot:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                with fig_plot:
                    st.write("no live data could be loaded")

    state.update_query_params()
    return state.get_state()


if __name__ == "__main__":
    main()
