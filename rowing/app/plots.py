
import streamlit as st 
import pandas as pd 
import plotly.express as px


def melt_livetracker_data(live_data, filter_distance=100):
    plot_data = live_data.dropna(
        subset=['distanceTravelled', "split", "avg split"]
    ).melt([
        "time", "distanceTravelled", "boat", "boatClass.DisplayName", "raceId"
    ])
    plot_data['crew'] = plot_data.boat + " " + plot_data['boatClass.DisplayName']
    
    facet_groups = live_data[
        (live_data.distanceTravelled > filter_distance)
        & (live_data.distanceTravelled < live_data.raceDistance)
    ].dropna(
        subset=["split", "avg split", "avg_speed"]
    ).melt([
        "time", "distanceTravelled", "raceBoatId"
    ]).groupby("variable")
    
    variable_types = facet_groups.value.first().map(type)
    facets = variable_types.index[variable_types != str]

    plot_data = plot_data.merge(
        live_data, on=['time', 'distanceTravelled', 'boat', 'raceId']
    )


    facet_max = facet_groups.value.max()[facets]
    facet_min = facet_groups.value.min()[facets]
    facet_ptp = facet_max - facet_min 
    facet_data = pd.concat(
        [facet_min - facet_ptp*0.1, facet_max + facet_ptp*0.1, ], 
    axis=1).apply(tuple, axis=1).rename("range").to_frame()

    facet_format = {f: True for f in facets}
    facet_format['kilometrePersSecond'] = False
    facet_format['avg_speed'] = ":0.1f"
    facet_format["distance from PGMT"] = ":0.1f"
    facet_format['PGMT'] = ':.1%'
    facet_format['split'] = "|%-M:%S.%L"
    facet_format['avg split'] = "|%-M:%S.%L"
    facet_format['ResultTime'] = "|%-M:%S.%L"

    facet_data['matches'] = None
    facet_data['title_text'] = facet_data.index
    facet_axes = facet_data.T.to_dict()

    for facet in facet_axes:
        fmt = facet_format[facet]
        if fmt is not True:
            facet_axes[facet]['tickformat'] = fmt

    facet_axes['PGMT']['tickformat'] = ',.1%'
    facet_axes['split']['tickformat'] = "%-M:%S"
    facet_axes['ResultTime']['tickformat'] = "%-M:%S"

    for col in ["distance from PGMT", "distanceFromLeader"]:
        facet_axes[col]['range'] =  facet_axes[col]['range'][::-1]

    return plot_data, facet_axes, facet_format

def make_livetracker_plot(
        facets, plot_data, facet_axes, facet_format, 
        facet_row_spacing=0.01, height=1000, width=800, **kwargs
    ):
    facet_rows = {facet: len(facets) - i for i, facet in enumerate(facets)}
    fig = px.line(
        plot_data[plot_data.variable.isin(facets)], 
        x = 'distanceTravelled', 
        y = "value", 
        color = 'crew', 
        hover_data=facet_format, 
        facet_row="variable", 
        category_orders={
            "variable": facets,
        }, 
        facet_row_spacing=facet_row_spacing,
        height=height, 
        width=width,
        **kwargs
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    fig.update_annotations(text="")
    for facet, row in facet_rows.items():
        fig.update_yaxes(row=row, **facet_axes[facet])

    return fig


# def plot_livedata(live_boat_data):
#     boat_data = live_boat_data.stack().reset_index(-1).sort_values("distanceTravelled")
#     boat_data['split'] = 500 / boat_data.metrePerSecond
#     boat_data['split min/500'] =  pd.Timestamp(0) + pd.to_timedelta(boat_data['split'], 's')

#     sel = boat_data.loc[
#         (boat_data.distanceTravelled > 100) & (boat_data.distanceTravelled < 2000),
#         boat_data.columns[boat_data.dtypes != boat_data.boat.dtype]
#     ]
#     sel_max = sel.max()
#     sel_min = sel.min()
#     sel_ptp = sel_max - sel_min

#     lims = pd.concat([
#         sel_min - sel_ptp * 0.1,  sel_max + sel_ptp * 0.1
#     ], axis=1).apply(tuple, axis=1)


#     gmt_fig = px.line(
#         boat_data, 
#         x=boat_data.distanceTravelled, 
#         y=boat_data.distanceFromLeader, 
#         hover_data=[
#             'distanceFromLeader', 'metrePerSecond', 'strokeRate',
#         ],
#         color='boat',
#         range_y=lims['distanceFromLeader']
#     )
#     st.plotly_chart(gmt_fig)

#     fig = px.line(
#         boat_data, 
#         x='distanceTravelled', 
#         y='split min/500', 
#         hover_data=[
#             'distanceFromLeader', 'metrePerSecond', 'strokeRate',
#         ],
#         color='boat',
#         range_y=lims['split min/500']
#     )
#     fig.update_layout(
#         yaxis = dict(
#             tickformat = "%M:%S.%f",   
#         ),
#     )
#     st.plotly_chart(fig)

#     fig = px.line(
#         boat_data, 
#         x='distanceTravelled', 
#         y='strokeRate', 
#         hover_data=[
#             'distanceFromLeader', 'metrePerSecond', 'strokeRate', 
#         ],
#         color='boat',
#         range_y=lims['strokeRate']
#     )
#     st.plotly_chart(fig)