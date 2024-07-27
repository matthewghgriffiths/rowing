
import streamlit as st

import logging

import sys
import os
from pathlib import Path

from tqdm.autonotebook import tqdm

DIRPATH = Path(__file__).resolve().parent
LIBPATH = str(DIRPATH.parent.parent.parent)
try:
    from rowing.app import select, state, plots
    from rowing.world_rowing import api
except ImportError:
    realpaths = [os.path.realpath(p) for p in sys.path]
    if LIBPATH not in realpaths:
        sys.path.append(LIBPATH)

    from rowing.app import select, state, plots
    from rowing.world_rowing import api


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

st.set_page_config(
    page_title="World Rowing Realtime Livetracker",
    layout='wide',
    # page_icon=":rowing:",
)


def main(params=None):
    state.update(params or {})
    st.title("World Rowing Realtime Livetracker")
    st.write(
        """
        Allows the following of the livetracker data from a race in realtime.

        If there are no live races currently on, then you can check _replay_ in the sidebar
        to see replays of historical races. 
        """
    )

    state.reset_button()

    with st.sidebar:
        with st.expander("Settings"):
            realtime_sleep = st.number_input(
                "poll", 0., 10., state.get("poll", 3.), step=0.5
            )
            replay = st.checkbox(
                "replay race data", state.get("replay", False))
            replay_step = st.number_input(
                "replay step", 1, 100, state.get("replay_step", 10))
            replay_start = st.number_input(
                "replay step", 0, 1000, state.get("replay_start", 0))

            fig_params = plots.select_figure_params()
            clear = st.button("clear cache")
            if clear:
                st.cache_data.clear()

    with st.expander("Last Races"):
        n_races = st.number_input(
            "Load how many races?", 0, value=0, step=1)
        if n_races > 0:
            races, race_boats, intermediates = select.last_race_results(
                n_races, fisa=False, cached=False)
            race_name = races['Boat Class'] + ": " + races['Phase']
            race_name.index = races.Race
            races['Race'] = races['Race'].replace(race_name)
            if not intermediates.empty:
                intermediates['Race'] = intermediates[
                    'Race'].replace(race_name)
                table = select.unstack_intermediates(intermediates)
                table.index.names = ['Race', 'Inter']
                race_order = races.sort_values("Race Start").Race
                order = race_order[
                    race_order.isin(table.index.levels[0])]

                st.dataframe(
                    table.loc[order],
                    height=(len(table) + 1) * 35 + 3,
                    use_container_width=True,
                    column_config={
                        "Race": st.column_config.TextColumn(
                            width="small",
                        ),
                        "Intermediate": st.column_config.TextColumn(
                            width="small",
                        )
                    }
                )

    kwargs = {}
    race_expander = st.expander("Select race", True)
    if replay:
        with race_expander:
            kwargs['select_race'], kwargs['races_container'], kwargs["competition_container"] = st.tabs([
                "Select Race", "Filter Races", "Select Competition",
            ])

    with race_expander:
        race = select.select_live_race(replay, **kwargs)
        st.subheader("Loading race: ")
        st.write(race)

    st.subheader("Livetracker")

    live_race = select.get_live_race_data(
        race.race_id,
        realtime_sleep=realtime_sleep,
        replay=replay,
        replay_step=replay_step,
        replay_start=replay_start,
    )
    show_intermediates = st.empty()
    completed = st.progress(0., "Distance completed")
    fig_plot = st.empty()

    pbar = tqdm(live_race.gen_data(
        live_race.update,
        plots.live_race_plot_data,
        plots.make_plots,
    ))
    for fig, *_ in pbar:
        pbar.set_postfix(distance=live_race.distance)
        completed.progress(
            live_race.distance / live_race.race_distance,
            f"Distance completed: {live_race.distance}m/{live_race.race_distance}m"
        )

        with show_intermediates:
            if not live_race.lane_info.empty:
                plots.show_lane_intermediates(
                    live_race.lane_info, live_race.intermediates)

        if fig is not None:
            with fig_plot:
                fig = plots.update_figure(fig, **fig_params)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("no live data could be loaded")

    select.wait_for_next_race(n=5)
    state.update_query_params()
    return state.get_state()


if __name__ == "__main__":
    main()
