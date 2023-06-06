

import logging

import streamlit as st
import plotly.express as px 
import pandas as pd

from rowing.world_rowing import api, utils
from rowing.app import state, inputs, select

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="PGMTs",
    layout='wide'
    # page_icon="👋",
)
st.title("PGMTs")

with st.expander("Select competition"):
    competition = select.select_competition()
    competition_id = competition.name
    competition_type = competition.WBTCompetitionType
    state.set("CompetitionId", competition_id)

    f"loading Results for {competition.DisplayName}, type: {competition_type}"

races = select.get_races(competition_id)
events = select.get_events(competition_id)
results = select.get_results(competition_id)
boat_classes = select.get_boat_classes()

with st.expander("Set GMTs"):
    comp_boat_classes = select.get_competition_boat_classes(competition_id)
    cbts = select.select_best_times(comp_boat_classes, competition_type)

    if cbts.empty:
        st.write("No GMTS selected")
        st.stop()

    gmts = select.set_gmts(cbts, competition_type)

merged_results = api.merge_competition_results(
    results, races, events, boat_classes, gmts)

with st.expander("Filter Races"):
    results = select.select_results(
        merged_results,
        default=["racePhase", "distance", "Rank"],
        racePhase=['Final'], 
        distance=[2000], 
        Rank=[1],
        key='results', 
    ).set_index("event")

results['crew'] = results['Country'] + " " + results['boatClass']

st.subheader("View PGMTs")
st.dataframe(results.style.format({"PGMT": "{:,.2%}"}))

name = f"{competition.DisplayName} results"
st.download_button(
    label=f"Download {name}.csv",
    data=inputs.df_to_csv(results),
    file_name=f'{name}.csv',
    mime='text/csv',
)

results['ResultTime'] = utils.read_times(results.ResultTime) + pd.Timestamp(0)

col1, col2, col3 = st.columns(3)
facets_axes = {}
facets_axes['ResultTime'] = {
    "tickformat": "%-M:%S"
} 
facets_axes['PGMT'] = {
    "tickformat": ",.0%"
} 
hover_data = {
    'race.Date': True,
    'ResultTime': "|%-M:%S.%L",
    'PGMT': ":.1%",
    'event': True,
    'Country': True,
    'Rank': True,
    'Lane': True
}
with col1:
    x = st.selectbox(
        "Plot x", 
        options=["race.Date", "PGMT", "ResultTime"], 
        index=0
    )
with col2:
    y = st.selectbox(
        "Plot y", 
        options=["race.Date", "PGMT", "ResultTime"], 
        index=1
    )
with col3:
    group = st.selectbox(
        "Label", 
        options=[
            "crew", "event", "Country", "ResultTime", "Rank", "Lane", "racePhase"
        ], 
        index=0
    )

fig = px.scatter(
    results.reset_index(), 
    x=x, 
    y=y, 
    color=group,
    hover_data=hover_data
)
fig.update_xaxes(**facets_axes.get(x, {}))
fig.update_yaxes(**facets_axes.get(y, {}))

st.plotly_chart(fig, use_container_width=True)

st.subheader("View Intermediates")

with st.expander("Filter Intermediates"):
    intermediate_results = select.select_results(
        merged_results,
        default=["racePhase"],
        racePhase=['Final'], 
        key='intermediate_results', 
        filters=False, 
    ).groupby([
        "race", "Country", "distance"
    ]).first().unstack(-1)

st.dataframe(intermediate_results['PGMT'].style.format("{:,.2%}"))


name = f"{competition.DisplayName} PGMT"
st.download_button(
    label=f"Download {name}.csv",
    data=inputs.df_to_csv(intermediate_results.PGMT),
    file_name=f'{name}.csv',
    mime='text/csv',
)

st.dataframe(intermediate_results['ResultTime'])

name = f"{competition.DisplayName} intermediates"
st.download_button(
    label=f"Download {name}.csv",
    data=inputs.df_to_csv(intermediate_results.ResultTime),
    file_name=f'{name}.csv',
    mime='text/csv',
)


if st.button("reset"):
    st.experimental_set_query_params()
    st.experimental_rerun()
else:
    state.update_query_params()
