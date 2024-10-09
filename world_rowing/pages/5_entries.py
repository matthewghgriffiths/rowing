import streamlit as st

import sys
import os
from pathlib import Path

import logging

import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Entries",
    layout='wide'
)

DIRPATH = Path(__file__).resolve().parent
LIBPATH = str(DIRPATH.parent.parent)

try:
    import _
except ModuleNotFoundError:
    pass
finally:
    from rowing.app import state, inputs, select, plots
    from rowing.world_rowing import fields, api


logger = logging.getLogger(__name__)


def main(params=None):

    state.update(params or {})

    st.title("Entries")
    st.write(
        """
        Summary of the entries to a competition
        """
    )

    with st.sidebar:
        height = int(st.number_input(
            "Table Height",
            min_value=100,
            value=1000,
            step=100
        ))
        inputs.clear_cache()

    with st.expander("Select Competition"):
        today = pd.Timestamp.today()
        competition = select.select_competition(
            start_date=today + pd.Timedelta("180d"),
            end_date=today - pd.Timedelta("180d"),
        )
        competition_id = competition.competition_id
        competition_type = competition.WBTCompetitionType
        state.set("CompetitionId", competition_id)
        st.write(
            f"loading Results for {competition.competition}, type: {competition_type}"
        )

    with st.spinner("Downloading entries"):
        comp_boat_athletes = select.get_entries(competition_id)

    if comp_boat_athletes is None:
        st.write(
            f"No events could be loaded for {competition.competition}")
        st.stop()

    event_entries = comp_boat_athletes.groupby('Event').apply(
        lambda data: data.Boat.drop_duplicates().reset_index(drop=True)
    ).unstack(fill_value='')
    event_entries.columns += 1

    st.subheader("Entries Summary")
    st.dataframe(
        event_entries
    )

    if not st.toggle("Entries By Event"):
        boat_athlete_pos = comp_boat_athletes.set_index(
            ['Event', "Boat", "Position"]
        ).Athlete.unstack(fill_value='')
        st.dataframe(
            boat_athlete_pos,
            column_config={
                p: st.column_config.TextColumn(
                    width='medium'
                )
                for p in boat_athlete_pos.columns
            },
            height=height
        )
    else:
        for event, event_boats in comp_boat_athletes.groupby('Event'):
            with st.expander(event, False):
                boat_athlete_pos = event_boats.set_index(
                    ["Boat", "Position"]
                ).Athlete.unstack(fill_value='')
                st.dataframe(
                    boat_athlete_pos,
                    use_container_width=True,
                    column_config={
                        p: st.column_config.TextColumn(
                            width='small'
                        )
                        for p in boat_athlete_pos.columns
                    }
                )


if __name__ == "__main__":
    main()
