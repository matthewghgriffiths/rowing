
import streamlit as st
import random
import re
import logging

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

import seaborn as sns

import streamlit.components.v1 as components
from streamlit_plotly_mapbox_events import plotly_mapbox_events
import gspread_pandas

from rowing.analysis import geodesy

logger = logging.getLogger(__file__)

EMBED_TEMPLATE = '<iframe {attrs}></iframe>'
urls = [
    'https://www.google.com/maps/embed?pb=!4v1740937700326!6m8!1m7!1so13J2hDBvzD2DEBvBitudA!2m2!1d51.47518037604937!2d-0.2242292696858267!3f346.3098504563404!4f-2.9170632522334046!5f0.7820865974627469',
    'https://www.google.com/maps/embed?pb=!4v1740993098607!6m8!1m7!1sy_Dn-AB_q2Cc0ZAS07RnPA!2m2!1d51.47555724203108!2d-0.2513615534092112!3f111.85247397332517!4f5.401017195468057!5f0.4000000000000002',
    'https://www.google.com/maps/embed?pb=!4v1740937764503!6m8!1m7!1sHzD_WPH8y6hZM4z16TpiVg!2m2!1d51.48556562963177!2d-0.2275685590965088!3f329.5361314675874!4f1.9134370118583774!5f0.7820865974627469',
    'https://www.google.com/maps/embed?pb=!4v1740993154633!6m8!1m7!1sZQ4ePFQqdQ4L0_e5naGy4g!2m2!1d51.47112647679729!2d-0.2645214922996735!3f77.35616795021275!4f0.6503664590120479!5f1.8309513899817027',
    'https://www.google.com/maps/embed?pb=!4v1740993219312!6m8!1m7!1sPRSR_0uu80XA7lL9YJjw2A!2m2!1d51.48858251972053!2d-0.2308372793755596!3f278.9311651677046!4f-1.529163553946404!5f1.9587109090973311',
    'https://www.google.com/maps/embed?pb=!4v1740993013759!6m8!1m7!1sGz76EHNj-usIpkEw_53HxA!2m2!1d51.47751722208024!2d-0.2506622751974344!3f358.8312926662331!4f-3.9092282295517435!5f0.7820865974627469',
    'https://www.google.com/maps/embed?pb=!4v1740993387477!6m8!1m7!1sQy5XlyEFw47Qh5FzGKC0CQ!2m2!1d51.47867245009478!2d-0.2253952031610886!3f253.19073894612148!4f-2.23110410197782!5f1.9587109090973311',
    'https://www.google.com/maps/embed?pb=!4v1740992989025!6m8!1m7!1sS4nvotTAxX3OfqpiCyXaZw!2m2!1d51.47273697908322!2d-0.2537821746250593!3f19.61499272671466!4f0.4419032345348626!5f0.7820865974627469',
    'https://www.google.com/maps/embed?pb=!4v1740937659682!6m8!1m7!1s7n-WTPtXECxAvFLXDrNsfw!2m2!1d51.47199119470677!2d-0.2211678208793883!3f315.0939529616098!4f-5.864996603455182!5f0.7820865974627469',
    'https://www.google.com/maps/embed?pb=!4v1740937811578!6m8!1m7!1sbMfomb_2aR3XMm-QKQL1rQ!2m2!1d51.47894321344205!2d-0.2501845319736254!3f328.53376420842056!4f4.735363193930297!5f0.7820865974627469',
    'https://www.google.com/maps/embed?pb=!4v1740937828663!6m8!1m7!1sARNMTaU6sXKEYCWDWY7pdQ!2m2!1d51.48780458749148!2d-0.2434770474349015!3f199.24679586930398!4f-0.447988281292254!5f0.7820865974627469',
    'https://www.google.com/maps/embed?pb=!4v1740937791623!6m8!1m7!1s_JOvQy1AcO2irHu0JPfTFg!2m2!1d51.48937565474303!2d-0.236029361578029!3f267.28461971146044!4f4.750520231606188!5f0.7820865974627469',
    'https://www.google.com/maps/embed?pb=!4v1740993045740!6m8!1m7!1sR3Di9l9x-KEtpZRJwY4TBA!2m2!1d51.48502796256749!2d-0.2477391471914139!3f302.7402791263707!4f-3.3762630236734594!5f0.4000000000000002'
]


@st.cache_data
def get_data():
    views = pd.Series(urls).rename('url').to_frame()
    views['latitude'] = views.url.str.extract(
        r"\!1d(-?[0-9]+\.[0-9]+)")[0].astype(float)
    views['longitude'] = views.url.str.extract(
        r"\!2d(-?[0-9]+\.[0-9]+)")[0].astype(float)

    positions = pd.read_csv("data/tideway.csv").drop(columns='bearing')

    dists = pd.DataFrame(
        geodesy.cdist_haversine_km(views, positions),
        index=views.url, columns=positions.index
    )
    closest_points = dists.idxmin(1).rename("closest").to_frame().join(
        positions, on='closest'
    )
    return positions, views, closest_points


def get_gspread_config():
    if 'gspread' in st.secrets:
        config = st.secrets.get(f"gspread")
    else:
        config = gspread_pandas.conf.get_config()

    return config


@st.cache_resource
def get_sheet():
    return gspread_pandas.Spread(
        "169p_UhJAUfKG62jt6L21AG8c5AXV5UV6jEOuBIY_u-4",
        sheet="Ordered",
        config=get_gspread_config()
    )


def get_responses():
    sheet = get_sheet()
    return sheet.sheet_to_df(start_row=2).reset_index().set_index('Name')


options = [f"Q{i + 1}" for i in range(13)]
option_colors = dict(zip(
    options, sns.color_palette('husl', n_colors=len(options))))


def to_rgba(r, g, b, a=1.):
    return "rgba({:.0%},{:.0%},{:.0%},{:.0%})".format(r, g, b, a)


def main(state=None):
    logger.info("Guessr")
    st.set_page_config(
        page_title="Tideway Guessr Responses",
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    st.title("Tideway Guessr Responses")

    with st.sidebar:
        height = st.number_input(
            "Default Figure Height",
            min_value=100,
            max_value=3000,
            value=800,
            step=50,
        )
        zoom = st.number_input(
            "Default Zoom",
            min_value=1,
            max_value=20,
            value=13,
            step=1,
        )
        error_scale = st.number_input(
            "Solution Bubble scale",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )
        map_style = st.selectbox(
            "map style",
            ["open-street-map", "carto-positron", "carto-darkmatter"],
            key='landmark map style'
        )

    responses = get_responses()
    responses_lat = responses.filter(like="Lat", axis=1).astype(float)
    responses_lon = responses.filter(like="Long", axis=1).astype(float)
    responses_lat.columns = responses_lat.columns.str.extract(
        "(Q[0-9]+)")[0].rename('question')
    responses_lon.columns = responses_lon.columns.str.extract(
        "(Q[0-9]+)")[0].rename('question')
    # responses_lat.columns.name = 'question'
    # responses_lon.columns.name = 'question'
    responses_positions = pd.concat({
        'latitude': responses_lat,
        'longitude': responses_lon
    }, axis=1).stack(future_stack=True)
    lat, lon = responses_positions.mean()
    players = responses_positions.index.levels[0].difference(['Solution'])
    plot_order = sum(([p, 'Solution'] for p in players), [])

    fig = go.Figure()
    for q in options:
        qpos = responses_positions.xs(q, level=1)
        qsol = qpos.loc['Solution']
        error = (geodesy.haversine_km(qpos, qsol) * 1000).astype(int)
        qdraw = qpos.loc[plot_order].join(error.rename('error'))
        qdraw['label'] = qdraw.apply(
            lambda ans: q if ans.name == 'Solution'
            else f"{ans.name.strip()}, {ans.error:.0f}m",
            axis=1
        )
        fig.add_trace(go.Scattermapbox(
            lat=[qsol.latitude],
            lon=[qsol.longitude],
            mode='markers',
            marker=dict(
                size=max(20, error.mean() / error_scale),
                color=to_rgba(*option_colors[q], 0.5)
            ),
            name=f"{q}",
            legendgroup='Solutions',
            legendgrouptitle_text='Solutions',
            textposition='bottom right',
        ))
        fig.add_trace(go.Scattermapbox(
            lat=qdraw.latitude,
            lon=qdraw.longitude,
            text=qdraw['label'],
            mode='lines+markers',
            marker=dict(
                size=10, color=to_rgba(*option_colors[q], 0.5)
            ),
            line=dict(color=to_rgba(*option_colors[q], 0.8)),
            name=f"{q}",
            legendgroup='Guesses',
            legendgrouptitle_text='Guesses',
            textposition='bottom right',
            hovertemplate=(
                "<b>%{text}</b><br>"
                + "longitude: %{lon:.6f}<br>"
                + "latitude: %{lat:.6f}<br>"
                # + "altitude: %{customdata[0]:.0f}<br>"+ "ppm: %{marker.color:.2f}<extra></extra>"
            )
        ))

    fig.update_layout(
        {"uirevision": True},
        mapbox={
            'style': map_style,
            'center': {'lon': lon, 'lat': lat},
            'zoom': zoom
        },
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            # itemsizing='constant',
            groupclick="toggleitem",
            itemdoubleclick='toggleothers',
        ),
        height=height,
        overwrite=True,
        autosize=True,
        margin=dict(
            b=0, l=0, r=0, t=0,
            pad=10,
            autoexpand=True,
        )
    )
    st.plotly_chart(fig)

    # for i, url in enumerate(urls):
    #     st.header(f"Q{i + 1} Location")
    #     info = pd.concat({
    #         "streetview": views.loc[i, ['latitude', 'longitude']],
    #         "solution": closest_points.loc[url, ['latitude', 'longitude']],
    #     }).unstack()  # .droplevel(1)
    #     st.dataframe(
    #         info,
    #         column_config=dict(
    #             latitude=st.column_config.NumberColumn(format="%.7f"),
    #             longitude=st.column_config.NumberColumn(format="%.7f")
    #         )
    #     )

    #     attrs = dict(
    #         src=url,
    #         width="100%",
    #         height=f"{height}",
    #         style="border:0; ",
    #         allowfullscreen="",
    #         loading="lazy",
    #         referrerpolicy="no-referrer-when-downgrade",
    #     )

    #     embed_str = EMBED_TEMPLATE.format(
    #         attrs=" ".join(f'{k}="{v}"' for k, v in attrs.items()))
    #     components.html(embed_str, height=height)


if __name__ == "__main__":
    main()
