

import logging

import streamlit as st


from rowing.world_rowing import api, utils
from rowing.app import state, inputs, select


logger = logging.getLogger(__name__)

st.subheader("Select competition")

competition = select.select_competition()
competition_id = competition.name
competition_type = competition.WBTCompetitionType

state.set("CompetitionId", competition_id)

f"loading Results for {competition.DisplayName}, type: {competition_type}"

races = select.get_races(competition_id)
events = select.get_events(competition_id)
results = select.get_results(competition_id)
boat_classes = select.get_boat_classes()

st.subheader("Set GMT")

comp_boat_classes = select.get_competition_boat_classes(competition_id)
cbts = select.select_best_times(comp_boat_classes, competition_type)

if cbts.empty:
    st.write("No GMTS selected")
    st.stop()

gmts = select.set_gmts(cbts, competition_type)

st.subheader("View PGMTs")

merged_results = api.merge_competition_results(
    results, races, events, boat_classes, gmts)
results = select.select_results(
    merged_results,
    options=["racePhase", "distance", "Rank"],
    racePhase=['Final'], 
    distance=[2000], 
    Rank=[1],
    key='results', 
)
st.dataframe(results.style.format({"PGMT": "{:,.2%}"}))

st.subheader("View Intemediates")

intermediate_results = select.select_results(
    merged_results,
    options=["racePhase"],
    racePhase=['Final'], 
    key='intermediate_results', 
    filters=False, 
).groupby([
    "race", "Country", "distance"
]).first().unstack(-1)

st.dataframe(intermediate_results['PGMT'].style.format("{:,.2%}"))
st.dataframe(intermediate_results['ResultTime'])


if st.button("reset"):
    st.experimental_set_query_params()
    st.experimental_rerun()
else:
    state.update_query_params()
