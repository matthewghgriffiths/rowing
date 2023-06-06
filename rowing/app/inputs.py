
import logging
import streamlit as st
from io import StringIO

import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from . import state

logger = logging.getLogger(__name__)

def modal_button(label1, label2, key=None, mode=False):
    key = key or ".".join(["modal_button", label1, label2])

    logger.debug("modal_button: %s mode=%s", key, mode)

    mode = state.get(key, mode)
    container = st.empty()
    if mode:
        key1 = ".".join([key, label1])
        rerun = container.button(label1, key=key1)
        mode = not rerun
    else:
        key2 = ".".join([key, label2])
        rerun = mode = container.button(label2, key=key2)

    state.set(key, mode)
    st.session_state[key] = mode
    if rerun:
        state.update_query_params()
        st.experimental_rerun()

    return mode



def filter_dataframe(
        df: pd.DataFrame, options=None, default=None, key=None, categories=(), filters=True,
        select=True, select_col='select', **kwargs
) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    logger.debug("filter_dataframe: %s", key)
    model_key = f"{key}.modal"
    modify = modal_button("Remove Filters", "Add filters", key=model_key, mode=filters)

    if not modify:
        return df

    df = df.copy()
    modification_container = st.container()
    with modification_container:
        column_options = options or df.columns
        to_filter_columns = st.multiselect(
            "Filter dataframe on", 
            column_options, 
            default, 
            key=f"{key}.filter_columns"
        )
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            col_key = f"{key}.{column}"
            # Treat columns with < 10 unique values as categorical
            categorical = (
                is_categorical_dtype(df[column]) 
                or df[column].nunique() < 10
                or column in categories
            )
            if is_datetime64_any_dtype(df[column]):
                logger.debug("filter_dataframe: %s: datetime", col_key)
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=state.get(
                        col_key, (df[column].min(), df[column].max())
                    ),
                    key=f"{key}.{column}",
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            elif categorical:
                logger.debug("filter_dataframe: %s: categorical", col_key)
                options = df[column].unique()
                default = set(options).intersection(
                    state.get(col_key, kwargs.get(column, []))
                )
                logger.debug(
                    "filter_dataframe: %s: options=%r default=%s", 
                    col_key, default, options
                )
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    options,
                    default=default or None,
                    key=col_key
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                logger.debug("filter_dataframe: %s: number", col_key)
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=state.get(col_key, (_min, _max)),
                    step=step,
                    key=f"{key}.{column}"
                )
                df = df[df[column].between(*user_num_input)]
            else:
                logger.debug("filter_dataframe: %s: text", col_key)
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                    value=state.get(col_key, ""),
                    key=col_key,
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    if select:
        df[select_col] = st.checkbox("Select all", value=True, key=f"{key}.select_all")
        df = df.loc[
            df.index[
                st.experimental_data_editor(
                    df[[select_col] + list(column_options)].set_index(select_col)
                ).index.values
            ]
        ]

    return df



@st.cache_data
def df_to_csv(df, **kwargs):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(**kwargs).encode('utf-8')


def download_csv(df, name='data', button="Create download", key=None, **kwargs):
    with st.empty():
        if st.button(button, key=key):
            data = df_to_csv(df, **kwargs)
            st.download_button(
                label=f"Download {name}.csv",
                data=data,
                file_name=f'{name}.csv',
                mime='text/csv',
            )

def upload_csv(name="Upload csv", encoding="utf-8", key=None, **kwargs):
    uploaded = st.file_uploader(name, key=key)
    if uploaded:
        raw = StringIO(uploaded.getvalue().decode(encoding))
        df = pd.read_csv(raw, **kwargs)
        return df