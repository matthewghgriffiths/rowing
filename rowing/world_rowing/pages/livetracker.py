
import streamlit as st

try:
    import about
except ModuleNotFoundError:
    pass
finally:
    from rowing.app import select, inputs, state, plots
    from rowing.world_rowing import fields


# st.set_page_config(
#     page_title="World Rowing livetracker", layout='wide')


def main(params=None):
    st.session_state.update(params or {})

    st.title("World Rowing livetracker")

    st.write(
        """
        Allows loading, filtering and visualisation of livetracker data from a FISA competition.

        The livetracker data does not come with any time information, 
        except for the intermediate times, so the app estimates the race time 
        for each time for each timepoint to match as closely as possible the intermediate
        and final times.

        From these estimated times the app can calculate distance from PGMT, 
        which is the distance behind (or ahead) from a boat going at an even 
        percentage gold medal pace. 
        The percentage of this pace can be set in the app, and defaults to 100%. 
        It can be useful to set this PGMT of the pace boat lower to better visualise the race
        profile when the conditions are slower.
        
        Use the 'Select livetracker data' to select which competition and races you want to view. 
        You can select multiple races to view from the same competition at the same time.
        """
    )

    with st.sidebar:
        download = st.checkbox("automatically load livetracker data", True)
        with st.expander("Settings"):
            fig_params = plots.select_figure_params()

            threads = st.number_input(
                "number of threads to use", min_value=1, max_value=20,
                value=4,
                step=1
            )
            threads = int(threads)
            inputs.clear_cache()

    # st.subheader("Select livetracker data")
    with st.expander("Select livetracker data"):
        st.write(
            """
            Select which competition and races to visualise race profiles for.

            The most recent FISA competition will be loaded by default, 
            'select other competition' will allow you to choose older competitions. 

            To filter races, select which criteria to filter the races on using 'Filter Dataframe on', 
            for example, Event, Phase, Day or Boat Class. 
            """
        )
        select_competition, filter_races, select_gmts, filter_live = st.tabs([
            "Select Competition", "Filter Races", "Select GMTS", "Filter livetracker data"
        ])

    races = select.select_races(
        competition_container=select_competition,
        races_container=filter_races,
        filters=True, select_all=False, select_first=True,
        default=[
            # fields.Phase,
            # fields.Gender,
            fields.race_raceStatus
        ],
        **{
            #     fields.Phase: ['Final A'],
            fields.race_raceStatus: ["Official", "Unofficial"],
        }
    ).reset_index(drop=True)

    if races.empty:
        if st.session_state.get("expander.filter_races", False):
            st.caption("select races to load")
            st.stop()

        st.session_state["expander.filter_races"] = True
        st.rerun()
        raise st.runtime.scriptrunner.StopException()

    competition_id = races[fields.race_event_competitionId].iloc[0]

    with select_gmts:
        gmts = select.set_competition_gmts(competition_id)
        races = races.set_index("race_id").join(
            gmts.rename(fields.GMT), on=fields.boatClass)

    if not download:
        st.caption(f"Selected {len(races)} races")
        st.caption(
            "Checkbox 'load livetracker data' in sidebar to view race data")
        st.stop()

    with st.spinner("Downloading livetracker data"), st.empty():
        live_data, intermediates, lane_info = select.get_races_livedata(
            races, max_workers=threads)

    if live_data.empty:
        return state.get_state()

    return show_livetracker(live_data, fig_params, filter_live)


# @st.fragment
def show_livetracker(live_data, fig_params, filter_container=None):
    filter_container = filter_container or st.container
    with filter_container:
        live_data = select.filter_livetracker(live_data)

    if live_data.empty:
        return state.get_state()

    st.subheader("Show livetracker")

    live_data, PGMT = select.set_livetracker_paceboat(live_data)
    # live_data[]
    print(live_data.columns)
    col = f"Distance from {PGMT:.1%} PGMT"
    live_data[col] = live_data[fields.distance_from_paceboat]
    facets = [
        fields.PGMT,
        col,
        fields.split,
        fields.live_raceBoatTracker_strokeRate,
    ]

    with st.spinner("Generating livetracker plot"):
        plot_data, facet_axes, facet_format = plots.melt_livetracker_times(
            live_data, 100)
        facet_axes[col]['range'] = facet_axes[col]['range'][::-1]
        facet_format[col] = facet_format[fields.distance_from_paceboat]
        fig = plots.make_livetracker_plot(
            facets, plot_data, facet_axes, facet_format)
        fig = plots.update_figure(fig, **fig_params)
        st.plotly_chart(fig, use_container_width=True)

    state.reset_button()
    return state.get_state()


if __name__ == "__main__":
    main()
