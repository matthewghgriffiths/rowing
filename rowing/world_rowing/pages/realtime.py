

import time
from matplotlib import use
import streamlit as st

import logging

from tqdm.autonotebook import tqdm

# st.set_page_config(
# page_title="World Rowing Realtime Livetracker", layout='wide')

try:
    import about
except ModuleNotFoundError:
    pass
finally:
    from rowing.world_rowing import api
    from rowing.app import select, state, plots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def main(params=None):
    st.session_state.update(params or {})
    st.title("World Rowing Realtime Livetracker")
    st.write(
        """
        Allows the following of the livetracker data from a race in realtime.

        If there are no live races currently on, then you can check _replay_ in the sidebar
        to see replays of historical races. 
        """
    )

    with st.sidebar:
        with st.expander("Settings"):
            realtime_sleep = st.number_input("poll", 0., 10., 3., step=0.5)
            replay = st.checkbox(
                "replay race data",
                st.session_state.get("replay_race", False),
                key='replay_race'
            )
            replay_step = st.number_input("replay step", 1, 100, 10)
            replay_start = st.number_input("replay step", 0, 1000, 0)

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
            race_order = races.sort_values("Race Start").Race
            order = race_order[race_order.isin(
                intermediates['Race'])].drop_duplicates()
            race_lanes = intermediates.groupby([
                "Race", "Lane"
            ])['Boat'].first().sort_index()
            race_inters = intermediates.groupby([
                "Race", "Distance", "Boat"
            ])['ResultTime'].first()
            for race in order:
                st.write(f"##### {race}")
                lane_order = race_lanes.loc[race]
                inters = race_inters.loc[race].unstack()[lane_order]
                plots.show_intermediates(inters)

    kwargs = {}
    race_expander = st.expander("Select race", True)
    if replay:
        with race_expander:
            kwargs['select_race'], kwargs['races_container'], kwargs["competition_container"] = st.tabs([
                "Select Race", "Filter Races", "Select Competition",
            ])

    with race_expander:
        race = select.select_live_race(replay, **kwargs)
        if st.toggle("## Race details", True):
            st.dataframe(race, use_container_width=True)

        if st.toggle("## Crew lists", True):
            st.dataframe(select.get_crewlist(race.race_id))

        boat_class = api.BOATCLASSES.get(race.race_event_boatClassId)
        if st.toggle("## World Best Time", True):
            # st.subheader("World Best Time:")
            wbts = select.get_cbts(
                [boat_class]
            ).sort_values('Best Time')
            if boat_class:
                st.markdown("#### Details:")
                st.dataframe(
                    select.fields.to_streamlit_dataframe(wbts).T,
                    # use_container_width=True
                )

            fastest_id = wbts.loc[wbts['Best Time'].idxmin(),
                                  'bestTimes_RaceId']
            inters = select.get_race_intermediates(fastest_id)
            if not inters.empty:
                st.markdown("#### Intermediates:")
                plots.show_intermediates(
                    inters.ResultTime, use_container_width=True)

    state.reset_button()

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

    if "distance_completed" in st.session_state:
        last_dist = st.session_state["distance_completed"]
        last_time = st.session_state["last_time_update"]
    else:
        last_dist = st.session_state["distance_completed"] = 0
        last_time = st.session_state["last_time_update"] = time.time()

    fig_plot = st.empty()

    pbar = tqdm(live_race.gen_data(
        live_race.update,
        plots.live_race_plot_data,
        plots.make_plots,
    ))
    for fig, *_ in pbar:
        pbar.set_postfix(distance=live_race.distance)
        if live_race.distance == last_dist:
            elapsed = time.time() - last_time
        else:
            st.session_state["distance_completed"] = last_dist = live_race.distance
            st.session_state["last_time_update"] = last_time = time.time()
            elapsed = 0

        completed.progress(
            live_race.distance / live_race.race_distance,
            f"Distance completed: {live_race.distance}m/{live_race.race_distance}m, "
            f"{elapsed:0.1f}s out of date"
        )

        with show_intermediates:
            if not live_race.lane_info.empty:
                plots.show_lane_intermediates(
                    live_race.lane_info, live_race.intermediates)

        with fig_plot:
            if fig is not None:
                fig = plots.update_figure(fig, **fig_params)
                st.plotly_chart(
                    fig, use_container_width=True,
                    key=time.time()
                )
            else:
                st.write("no live data could be loaded")

    select.wait_for_next_race(n=5)
    # state.update_query_params()
    return state.get_state()


if __name__ == "__main__":
    main()
