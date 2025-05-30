
import streamlit as st

import logging

import plotly.express as px
import pandas as pd


# st.set_page_config(page_title="PGMTs", layout='wide')

logger = logging.getLogger(__name__)

try:
    import about
except ModuleNotFoundError:
    pass
finally:
    from rowing.app import state, inputs, select, plots
    from rowing.world_rowing import fields


def main(params=None):
    st.session_state.update(params or {})

    st.title("PGMTs")
    st.write(
        """
        Allows loading, filtering and visualisation of results and PGMTs from a FISA competition.
        """
    )

    with st.sidebar:
        with st.expander("Settings"):
            fig_params = plots.select_figure_params()
            inputs.clear_cache()

    with st.expander("Select Results"):
        select_competitions, set_gmts, filter_results = st.tabs([
            "Select competition", "Set GMTs", "Filter Results"
        ])

    with select_competitions:
        competition = select.select_competition()
        competition_id = competition.competition_id
        competition_type = competition.WBTCompetitionType
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
                fields.Distance,
                # fields.raceBoatIntermediates_Rank
            ],
            **{
                fields.Phase: ['Final A'],
                fields.Distance: [2000],
                fields.raceBoatIntermediates_Rank: [1],
            },
            key='results',
        )

        if st.button("Reload results"):
            st.cache_data.clear()
            select.api.clear_cache()

    if results is None:
        st.write("No results could be loaded")
        st.stop()
        raise st.runtime.scriptrunner.StopException()

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
    results[fields.GMT] = results[fields.GMT] + pd.Timestamp(0)

    plot_inputs = inputs.set_plotly_inputs(
        results.reset_index(),
        x=fields.race_Date,
        y=fields.PGMT,
        color=fields.BoatClass,
        facet_col=fields.Day,
        symbol=fields.raceBoats,
    )
    facets_axes = {
        fields.race_Date: {"matches": None},
        fields.ResultTime: {"tickformat": "%-M:%S"},
        fields.PGMT: {"tickformat": ",.0%"}
    }
    plot_inputs['hover_data'] = {
        fields.raceBoatIntermediates_Rank: True,
        fields.raceBoatIntermediates_ResultTime: "|%-M:%S.%L",
        fields.GMT: "|%-M:%S.%L",
    }
    plot_inputs['symbol_sequence'] = plots.SYMBOLS
    fig = px.scatter(**plot_inputs)
    fig_params['xaxes'] = facets_axes.get(plot_inputs['x'], {})
    fig_params['yaxes'] = facets_axes.get(plot_inputs['y'], {})
    fig_params['layout'].pop("legend")
    fig = plots.update_figure(fig, **fig_params)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("View Intermediates")

    with st.expander("Filter Intermediate Results"):
        res = select.select_competition_results(
            competition_id, gmts,
            default=[fields.Phase,],
            Phase=['Final A'],
            filters=False,
            key='intermediate_results'
        )
        if res is None:
            st.write("No results could be loaded")
            st.stop()
            raise st.runtime.scriptrunner.StopException()

        intermediate_results = res.groupby([
            fields.Race, fields.raceBoats, fields.Distance
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
        fields.Event: '{:}',
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

    return state.get_state()


if __name__ == "__main__":
    main()
