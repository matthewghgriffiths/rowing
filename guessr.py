
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

color_discrete_sequence = [
    '#636efa',
    '#EF553B',
    '#00cc96',
    '#ab63fa',
    '#FFA15A',
    '#19d3f3',
    '#FF6692',
    '#B6E880',
    '#FF97FF',
    '#FECB52',
]
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


def to_rgb(r, g, b):
    return "rgb({:.0%},{:.0%},{:.0%})".format(r, g, b)


def to_rgba(r, g, b, a=1.):
    return "rgba({:.0%},{:.0%},{:.0%},{:.0%})".format(r, g, b, a)


n_points = len(urls)
options = [f"Q{i + 1}" for i in range(n_points)]
option_colors = dict(zip(
    options, sns.color_palette('husl', n_colors=len(options))))
option_markers = {
    option: dict(
        size=10, color=to_rgba(*color),
    )
    for option, color in option_colors.items()
}
order = list(range(n_points))

KEYS_gspread_config = [
    'type', 'project_id', 'private_key_id', 'private_key',
    'client_email', 'client_id', 'auth_uri',
    'token_uri', 'auth_provider_x509_cert_url',
    'client_x509_cert_url'
]


def get_gspread_config():
    if 'gspread' in st.secrets:
        config = st.secrets.get(f"gspread")
    else:
        config = gspread_pandas.conf.get_config()

    return config


@st.cache_resource
def get_gspread_sheet():
    client = gspread_pandas.Client(
        config=get_gspread_config())
    spread = client.open_by_key(
        "169p_UhJAUfKG62jt6L21AG8c5AXV5UV6jEOuBIY_u-4")
    sheet = spread.get_worksheet_by_id(0)
    return sheet


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
    return positions, closest_points


def main(state=None):
    logger.info("Guessr")
    st.set_page_config(
        page_title="Tideway Guessr",
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    st.title("Tideway Guessr")

    name = st.text_input(
        "Name: ",
        autocomplete="name"
    )
    club = st.text_input(
        "Club (optional)",
        # autocomplete="email"
    )
    email = st.text_input(
        "Email (optional)",
        autocomplete="email"
    )
    positions, closest_points = get_data()
    lat, lon = positions[['latitude', 'longitude']].mean()
    completed = int(st.query_params.get('completed', 0))

    with st.sidebar:
        map_style = 'open-street-map'
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
            value=12,
            step=1,
        )
        map_style = st.selectbox(
            "map style",
            ["open-street-map", "carto-positron", "carto-darkmatter"],
            key='landmark map style'
        )

    layout = dict(
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

    # order = st.query_params.get_all('order')
    # if not order:
    #     random.shuffle(order := list(range(n_points)))
    #     st.query_params['order'] = order
    # else:
    #     order = [int(i) for i in order]

    url_order = [urls[i] for i in order]
    answers = {
        q: st.query_params.get(q) for q in options}
    show_options = [
        q if answers[q] else f"**{q}**" for q in options]

    results = st.container()
    # segmented, submit, reset = st.columns((4, 1, 1))
    segmented, submit, reset = [st.container() for _ in range(3)]
    col1, col2 = st.columns(2)
    segmented = submit = col2

    with segmented:
        current = int(st.query_params.get('current', 0))
        question = st.segmented_control(
            "Choose location",
            show_options,
            default=show_options[current],
        )
        if question:
            st.query_params['current'] = i = show_options.index(question)

    if all(answers.values()):
        with submit:
            answer_positions = pd.concat({
                q: positions.loc[[int(a)]]
                for q, a in answers.items()
            }, names=['question'])
            answer_positions['url'] = url_order
            answer_positions = answer_positions.reset_index(
                'question'
            ).set_index('url')

            compare = pd.concat({
                "answers": answer_positions,
                "correct": closest_points.drop(columns='closest'),
            }, axis=1).set_index(
                [('answers', 'question')]
            ).rename_axis(index='question')

            compare[('error', 'm')] = geodesy.haversine_km(
                compare.answers, compare.correct
            ).round(3) * 1000

            if not name:
                st.write("Please enter your name to submit")
            elif completed or st.button(
                "Submit answers", type='primary'
            ):
                st.query_params['completed'] = completed = 1

    else:
        st.query_params['completed'] = completed = 0
        unanswered_qs = [q for q, a in answers.items() if not a]
        unanswered = ", ".join(unanswered_qs)
        with submit:
            n_answered = n_points - len(unanswered_qs)
            progress = n_answered / n_points
            st.progress(
                progress,
                f"Answered {n_answered} / {n_points} locations, "
                f"unanswered question(s): {unanswered}"
            )

    if completed:
        answers = pd.concat([
            compare.answers.stack().sort_index(),
            compare.error.stack().sort_index()
        ])
        hash_value = pd.util.hash_pandas_object(answers)
        print(hash_value)
        info = pd.concat({
            'info': pd.Series(dict(
                submitted=str(pd.Timestamp.now()),
                name=name,
                club=club,
                email=email,
            ))
        })
        to_submit = pd.concat([info, answers])
        vals = tuple(to_submit)[1:]
        submission_hash = hash(vals)
        try:
            if submission_hash != st.session_state.get('hash'):
                with st.spinner("Submitting Results"):
                    sheet = get_gspread_sheet()
                    sheet.insert_row(list(to_submit), index=3)
                st.session_state['hash'] = submission_hash
            else:
                print("Already submitted")
        except Exception as e:
            st.toast("Submission Failed, try again?")
            completed = False

        with results:
            st.dataframe(compare)
            avg_error = compare.error.m.mean()
            st.write(f"Average error = {avg_error:.1f}")

    with reset:
        if not completed:
            if st.toggle("Reset answers") and st.button(
                "Confirm: clicking this button will clear all progress",
                type='primary',
            ):
                st.query_params.clear()
                st.rerun()

    with col1:
        if question:
            i = show_options.index(question)
            url = url_order[i]

            st.header(f"{question} Location")
            st.write(
                "Click on the blue line on the map "
                "where you think the closest point "
                "to the picture below is"
            )

            attrs = dict(
                src=url,
                width="100%",
                height=f"{height + 700}",
                style="border:0; margin-top: -150px;",
                allowfullscreen="",
                loading="lazy",
                referrerpolicy="no-referrer-when-downgrade",
            )

            embed_str = EMBED_TEMPLATE.format(
                attrs=" ".join(f'{k}="{v}"' for k, v in attrs.items()))
            components.html(embed_str, height=height)
        else:
            st.write("Please pick a location")

    if completed:
        fig = go.Figure()
        for _, cmp in compare.iterrows():
            q = cmp.name
            answer = cmp.answers
            sol = cmp.correct
            error = cmp.error.m

            is_current = q == question

            marker = {
                **option_markers[q],
                'size': 30 if is_current else 5,
            }
            fig.add_trace(go.Scattermapbox(
                lat=[answer.latitude],
                lon=[answer.longitude],
                mode='markers',
                name=q,
                marker=option_markers[q],
                legendgroup='Answers',
                legendgrouptitle_text='Answers',
                textposition='bottom right',
            ))
            if is_current:
                marker = {
                    'color': to_rgba(*option_colors[q], 0.9),
                    'size': 30,
                }
                fig.add_trace(go.Scattermapbox(
                    lat=[sol.latitude],
                    lon=[sol.longitude],
                    mode='markers',
                    name=f"Actual: {q}",
                    marker=marker,
                    legendgroup='Answers',
                    legendgrouptitle_text='Answers',
                    textposition='bottom right',
                )
                )
            fig.add_trace(go.Scattermapbox(
                lat=[answer.latitude, sol.latitude],
                lon=[answer.longitude, sol.longitude],
                mode='lines',
                name=f"{q}, {error:.0f} m",
                marker=marker,
                legendgroup='Error',
                legendgrouptitle_text='Error',
                textposition='bottom right',
            ))

        fig.update_layout(**layout)
        with col2:
            st.plotly_chart(fig)

    else:
        fig = go.Figure()
        fig.add_trace(go.Scattermapbox(
            lat=positions.latitude,
            lon=positions.longitude,
            mode='lines',
            line_color=color_discrete_sequence[0],
            showlegend=False,
            textposition='bottom right',
        ))
        for q, answer in answers.items():
            if answer is not None:
                answer_pos = positions.loc[[int(answer)]]
                fig.add_trace(go.Scattermapbox(
                    lat=answer_pos.latitude,
                    lon=answer_pos.longitude,
                    mode='markers',
                    name=q,
                    marker=option_markers[q],
                    legendgroup='Answers',
                    legendgrouptitle_text='Answers',
                    textposition='bottom right',
                ))

        fig.update_layout(**layout)

        with col2:
            points, *_ = plotly_mapbox_events(fig)
            if points:
                point = points[0]
                index = point['pointNumber']
                if index != st.session_state.get('last_point'):
                    st.query_params[options[i]] = index
                    st.session_state['last_point'] = index
                    st.rerun()

    st.session_state['completed'] = int(completed)


if __name__ == "__main__":
    main()
