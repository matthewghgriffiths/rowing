
import logging
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm.autonotebook import tqdm
import streamlit as st

import plotly as pl
from plotly import express as px, subplots

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

    with st.sidebar:
        with st.expander("Settings"):
            dummy = st.checkbox("load dummy data", True)
            dummy_index = st.number_input("dummy_index", 0, 1000, 0)
            dummy_step = st.number_input(
                "dummy_step", 1, 100, state.get("dummy_step", 10))

    if dummy:
        races = select.select_races(
            filters=True, select_all=True
        )
        race = inputs.select_dataframe(races, fields.Race)
    else:
        race = api.get_live_race()
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
        realtime_sleep=0.1,
        dummy=dummy,
        dummy_index=dummy_index,
        dummy_step=dummy_step
    )

    fig_plot = st.empty()

    facets = [
        fields.live_raceBoatTracker_distanceFromLeader,
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

            if plot_data is not None:
                fig = px.line(
                    plot_data,
                    x=fields.live_raceBoatTracker_distanceTravelled,
                    y='value',
                    color=fields.raceBoats,
                    facet_row="live",
                    hover_data=facets
                )
                fig.update_yaxes(
                    matches=None
                )
                with fig_plot:
                    st.plotly_chart(fig)
            else:
                with fig_plot:
                    st.write("no live data could be loaded")

    live_race.tracker.realtime_history
    live_race.tracker.livetracker_history

    state.reset_button()
    state.update_query_params()
    return state.get_state()


if __name__ == "__main__":
    main()
