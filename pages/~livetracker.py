

import logging

import streamlit as st

import numpy as np
import pandas as pd

import plotly.express as px

from rowing.world_rowing import api, livetracker, utils
from rowing.app import select, inputs, state, plots

logger = logging.getLogger(__name__)

st.set_page_config(layout='wide')


st.title("Show livetracker data for races")

if inputs.modal_button(
    "input race", "select race", 
):
    competition = select.select_competition()
    competition_id = competition.name
    competition_type = competition.WBTCompetitionType
    races = select.select_races(competition_id)
    race = select.select_race(races)
    race_id = race['race.id']
    name = race['race.DisplayName']
    boat_class = race['boatClass.DisplayName']
    date = race['race.Date']
    event = race['race.event.DisplayName']
    st.write("race id:")
    st.write(f"`{race_id}`")
else:
    race_id = st.text_input(
        "enter race_id", 
        state.get("race_id") or api.get_most_recent_race().name
    )
    race = api.get_race(race_id)
    name = race.DisplayName
    boat_class = race['boatClass.DisplayName']
    date = race.Date 
    event = race['event.DisplayName']

state.set("race_id", race_id)


cbts = select.select_best_times([boat_class])
gmts = select.set_gmts(cbts)
gmt = gmts[race['boatClass.DisplayName']].total_seconds()
st.write(f"GMT set to {utils.format_totalseconds(gmt)}")

st.subheader(
    f"Live data for {name}, {event}"
)
live_data, live_boat_data, intermediates, distance = \
    select.get_livedata(race_id, gmt=gmt)

intermediates.index = intermediates.distance.fillna("ffill", axis=1).iloc[:, 0]
intermediate_times = intermediates.ResultTime.apply(lambda s: s.dt.total_seconds()).T

col1, col2, col3 = st.columns(3)
with col1:
    st.text("Time")
    st.dataframe(intermediates.ResultTime.applymap(utils.format_timedelta).T)
with col2:
    st.text("Difference")
    st.dataframe(intermediate_times - intermediate_times.values.min(0, keepdims=True))
with col3:
    st.text("Position")
    st.dataframe(intermediates.Rank.T)

plots.plot_livedata(live_data)

state.reset()
state.update_query_params()
