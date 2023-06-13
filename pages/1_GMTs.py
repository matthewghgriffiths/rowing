

import logging

import streamlit as st
import plotly.express as px 
import pandas as pd

from rowing.world_rowing import api, utils, fields
from rowing.app import state, inputs, select

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="PGMTs",
    layout='wide'
)
st.title("PGMTs")

with st.expander("Select competition"):
    competition = select.select_competition()
    competition_id = competition.competition_id
    competition_type = competition.WBTCompetitionType
    state.set("CompetitionId", competition_id)

    f"loading Results for {competition.competition}, type: {competition_type}"

with st.expander("Set GMTs"):
    gmts = select.set_competition_gmts(competition_id, competition_type)

with st.expander("Filter Results"):
    results = select.select_competition_results(
        competition_id, gmts,
        default=[
            fields.Phase, 
            fields.distance, 
            fields.raceBoatIntermediates_Rank
        ],
        **{
            fields.Phase: ['Final A'], 
            fields.distance: [2000], 
            fields.raceBoatIntermediates_Rank: [1],
        },
        key='results', 
    )

st.subheader("View PGMTs")

st.dataframe(fields.to_streamlit_dataframe(results))

name = f"{competition.competition} results"
st.download_button(
    label=f"Download {name}.csv",
    data=inputs.df_to_csv(results),
    file_name=f'{name}.csv',
    mime='text/csv',
)

results[fields.raceBoatIntermediates_ResultTime] = \
    results[fields.raceBoatIntermediates_ResultTime] + pd.Timestamp(0)

plot_inputs = inputs.set_plotly_inputs(
    results.reset_index(), 
    x=fields.race_Date, 
    y=fields.PGMT, 
    group=fields.crew, 
    facet_col=fields.Day, 
)
facets_axes = {
    fields.race_Date: {"matches": None},
    fields.ResultTime: {"tickformat": "%-M:%S"},
    fields.PGMT: {"tickformat": ",.0%"} 
}
# hover_data = {
#     'race_Date': True,
#     'raceBoatIntermediates_ResultTime': "|%-M:%S.%L",
#     'PGMT': ":.1%",
#     'event': True,
#     'raceBoats': True,
#     'raceBoatIntermediates_Rank': True,
#     'raceBoats_Lane': True
# }
fig = px.scatter(**plot_inputs)
fig.update_xaxes(**facets_axes.get(plot_inputs['x'], {}))
fig.update_yaxes(**facets_axes.get(plot_inputs['y'], {}))

st.plotly_chart(fig, use_container_width=True)

st.subheader("View Intermediates")

with st.expander("Filter Intermediate Results"):
    intermediate_results = select.select_competition_results(
        competition_id, gmts, 
        default=[fields.Phase,],
        Phase=['Final A'], 
        filters=False, 
        key='intermediate_results'
    ).groupby([
        fields.race, fields.raceBoats, fields.distance
    ]).first().unstack(-1)

name = f"{competition.competition} all intermediates"
st.download_button(
    label=f":inbox_tray: Download {name}.csv",
    data=inputs.df_to_csv(intermediate_results),
    file_name=f'{name}.csv',
    mime='text/csv',
)

col_formats = {
    fields.PGMT: '{:,.2%}',
    fields.raceBoatIntermediates_ResultTime: '{:}',
    fields.GMT: '{:}',
    fields.boatClass: '{:}',
    fields.Day: '{:}',
    fields.event: '{:}',
    fields.race_event_competition: '{:}',
    fields.Phase: '{:}',
    fields.raceBoatIntermediates_Rank: '{:}',
    fields.raceBoats_Lane: '{:}',
}
cols = st.multiselect(
    "select which data to show", 
    options=col_formats,
    default=[fields.PGMT, fields.raceBoatIntermediates_ResultTime]
)

for col in cols:
    st.dataframe(fields.to_streamlit_dataframe(intermediate_results[col]))

    name = f"{competition.competition} {col}"
    st.download_button(
        label=f":inbox_tray: Download {name}.csv",
        data=inputs.df_to_csv(intermediate_results[col]),
        file_name=f'{name}.csv',
        mime='text/csv',
    )


state.reset_button()
state.update_query_params()
