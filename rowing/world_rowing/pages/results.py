
import streamlit as st

import sys
import os
from pathlib import Path

import logging

import plotly.express as px
import pandas as pd

# st.set_page_config(
#     page_title="Results",
#     layout='wide'
# )

try:
    import about
except ModuleNotFoundError:
    pass
finally:
    from rowing.world_rowing import fields
    from rowing.app import state, inputs, select, plots


logger = logging.getLogger(__name__)


def main(params=None):

    state.update(params or {})

    st.title("Results")
    st.write(
        """
        Summary of the results of a competition 
        """
    )

    with st.sidebar:
        inputs.clear_cache()

    with st.expander("Select Results", expanded=True):
        set_tables, select_competitions, set_gmts, filter_results = st.tabs([
            'Set tables', "Select competition", "Set GMTs", "Filter Results",
        ])

    with select_competitions:
        competition = select.select_competition()
        competition_id = competition.competition_id
        competition_type = competition.WBTCompetitionType
        state.set("CompetitionId", competition_id)
        st.write(
            f"loading Results for {competition.competition}, type: {competition_type}"
        )

    with set_gmts:
        gmts = select.set_competition_gmts(competition_id, competition_type)

    with filter_results:
        results = select.select_competition_results(
            competition_id, gmts,
            default=[
                # fields.Phase,
                # fields.Distance,
                # fields.raceBoatIntermediates_Rank
            ],
            **{
                # fields.Phase: ['Final A'],
                # fields.Distance: [2000],
                # fields.raceBoatIntermediates_Rank: [1],
            },
            key='results',
        )
        full_results = fields.to_streamlit_dataframe(results)
        # full_results['PGMT'] = full_results['PGMT'].map("{:.2%}".format)

    with set_tables:
        column_order = st.selectbox(
            "Select column order",
            options=[
                "Intermediate Position", "Finish Position", "Lane"
            ],
        )
        show_values = st.multiselect(
            "Select what values to show",
            options=[
                'Intermediate Time', 'PGMT',
                "Intermediate Position", 'Finish Position',
                "Lane"
            ],
            default=[
                'Intermediate Time', 'PGMT'
            ]
        )
        order_by = st.multiselect(
            "Select which values to order results by",
            options=full_results.columns.difference(
                ["Race", "Distance", column_order]
            ),
            default=['Race Start']
        )
        ascending = st.checkbox(
            "Order in ascending order?"
        )
        groupby = st.multiselect(
            "Select how to group tables, "
            "clear to show all results as one table",
            options=full_results.columns.difference(
                ["Race", "Distance", column_order]
            ),
            default=['Day']
        )
        expand_all = st.toggle("expand all", False)

        if st.button("Reload results"):
            st.cache_data.clear()
            select.api.clear_cache()

    if groupby:
        groups = full_results.groupby(groupby)
        for key, key_results in groups:
            race_results = key_results.drop_duplicates(
                ["Race", "Distance", column_order]
            ).set_index(["Race", "Distance", column_order])
            race_results['PGMT'] = race_results['PGMT'].map("{:.2%}".format)

            table = race_results[
                ['Boat'] + show_values
            ].apply('<br>'.join, axis=1).unstack(
                fill_value=''
            )
            if order_by:
                order = key_results.groupby(
                    "Race"
                )[order_by].first().sort_values(
                    by=order_by,
                    ascending=ascending
                ).index
                table = table.loc[order]

            with st.expander(",".join(map(str, key)), expanded=expand_all):
                st.markdown(
                    table.style.to_html(
                        max_rows=None,
                    ),
                    unsafe_allow_html=True
                )
    else:
        with st.expander("All results", expand_all):
            race_results = full_results.drop_duplicates(
                ["Race", "Distance", column_order]
            ).set_index(["Race", "Distance", column_order])

            table = race_results[
                ['Boat'] + show_values
            ].apply('<br>'.join, axis=1).unstack(
                fill_value=''
            )
            if order_by:
                order = full_results.groupby(
                    "Race"
                )[order_by].first().sort_values(
                    by=order_by,
                    ascending=ascending
                ).index
                table = table.loc[order]

            st.markdown(
                table.style.to_html(
                    max_rows=None,
                ),
                unsafe_allow_html=True
            )

    with st.expander("Downloadable Results"):
        st.write(
            "Click the top left of the table to download "
            "the results as a csv"
        )
        st.dataframe(full_results)


if __name__ == "__main__":
    main()
