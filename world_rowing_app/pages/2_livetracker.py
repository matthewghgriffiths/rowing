import streamlit as st

import numpy as np
import pandas as pd

import plotly.express as px

from rowing.world_rowing import fields
from rowing.app import select, inputs, state, plots

st.set_page_config(
    page_title="World Rowing livetracker",
    layout='wide'
    # page_icon="👋",
)


def main(params=None):
    state.update(params or {})

    st.title("World Rowing livetracker")

    with st.sidebar:
        download = st.checkbox("automatically load livetracker data", True)
        with st.expander("Settings"):
            threads = st.number_input(
                "number of threads to use", min_value=1, max_value=20,
                value=state.get("threads", 6),
                step=1
            )
            threads = int(threads)
            state.set("threads", threads)

    st.subheader("Select livetracker data")

    races = select.select_races(
        filters=True, select_all=False, select_first=True,
        default=[
            fields.Phase,
            fields.Gender,
            fields.race_raceStatus
        ],
        **{
            fields.Phase: ['Final A'],
            fields.race_raceStatus: ["Official"],
        }
    ).reset_index(drop=True)

    if races.empty:
        if state.get("expander.filter_races", False):
            print(state.STATE)
            st.caption("select races to load")
            st.stop()

        state.set("expander.filter_races", True)
        state.update_query_params()
        st.experimental_rerun()

    competition_id = races[fields.race_event_competitionId].iloc[0]
    with st.expander("Select GMTs"):
        gmts = select.set_competition_gmts(competition_id)
        races = races.set_index("race_id").join(
            gmts.rename(fields.GMT), on=fields.boatClass)

    if not download:
        st.caption(f"Selected {len(races)} races")
        st.caption(
            "Checkbox 'load livetracker data' in sidebar to view race data")
        st.stop()

    with st.spinner("Downloading livetracker data"), st.empty():
        live_data, intermediates = select.get_races_livedata(
            races, max_workers=threads)

    with st.expander("Filter livetracker data"):
        live_data = select.filter_livetracker(live_data)

    st.subheader("Show livetracker")

    live_data, PGMT = select.set_livetracker_PGMT(live_data)
    facets = [
        fields.PGMT,
        fields.distance_from_pace,
        fields.split,
        fields.live_raceBoatTracker_strokeRate,
    ]

    with st.spinner("Generating livetracker plot"):
        plot_data, facet_axes, facet_format = args = \
            plots.melt_livetracker_data(live_data, 100)
        fig = plots.make_livetracker_plot(
            facets, *args,
        )
        st.plotly_chart(fig, use_container_width=True)

    state.reset_button()
    state.update_query_params()
    return state.get_state()


if __name__ == "__main__":
    main()