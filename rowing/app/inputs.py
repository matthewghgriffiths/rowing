
import logging
from matplotlib import category
import streamlit as st
from io import StringIO

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from ..world_rowing import fields, api

logger = logging.getLogger(__name__)


def get_url():
    import urllib.parse
    try:
        sess_mgr = st.runtime.get_instance()._session_mgr
        sessions = sess_mgr.list_active_sessions()
        req = sessions[0].client.request

        joinme = (req.protocol, req.host, "", "", "", "")
        my_url = urllib.parse.urlunparse(joinme)
    except RuntimeError:
        my_url = 'localhost'

    return my_url


def clear_cache():
    clear = st.button("clear cache")
    if clear:
        st.cache_data.clear()
        api.clear_cache()


def modal_button(label1, label2, key=None, mode=False):
    key = key or ".".join(["modal_button", label1, label2])

    logger.debug("modal_button: %s mode=%s", key, mode)

    container = st.empty()
    if mode:
        key1 = ".".join([key, label1])
        rerun = container.button(label1, key=key1)
        mode = not rerun
    else:
        key2 = ".".join([key, label2])
        rerun = mode = container.button(label2, key=key2)

    st.session_state[key] = mode
    if rerun:
        st.rerun()

    return mode


def filter_dataframe(
        df: pd.DataFrame, options=None, default=None, key=None, categories=(), filters=True,
        select=True, select_col='select', select_all=True, select_first=False,
        column_order=None,  column_config=None, num_rows='fixed', use_container_width=False,
        disabled=False, modification_container=None,
        **kwargs
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
    modification_container = modification_container or st.container()
    with modification_container:
        modify = modal_button("Remove Filters", "Add filters",
                              key=model_key, mode=filters)

    if not modify:
        st.dataframe(
            df,
            column_order=column_order,
            column_config=column_config,
            use_container_width=use_container_width,
        )
        return df

    with modification_container:
        df = df.copy()
        column_options = options or column_order or df.columns

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
                    value=(df[column].min(), df[column].max()),
                    key=f"{key}.{column}",
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(
                        map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            elif categorical:
                logger.debug("filter_dataframe: %s: categorical", col_key)
                options = df[column].unique()
                default = set(options).intersection(kwargs.get(column, []))
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
                    value=(_min, _max),
                    step=step,
                    key=f"{key}.{column}"
                )
                df = df[df[column].between(*user_num_input)]
            else:
                logger.debug("filter_dataframe: %s: text", col_key)
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                    key=col_key,
                )
                if user_text_input:
                    df = df[df[column].astype(
                        str).str.contains(user_text_input)]

    if select and not df.empty:
        with modification_container:
            df[select_col] = st.checkbox(
                "Select all", value=select_all, key=f"{key}.select_all")

        if select_first:
            df[select_col] = np.r_[True, df[select_col].iloc[1:]]

        sel_df = st.data_editor(
            df[[select_col] + list(column_options)].set_index(select_col),
            column_order=column_order,
            column_config=column_config,
            num_rows=num_rows,
            use_container_width=use_container_width,
            disabled=disabled
        )
        df = df.copy()
        sel_index = df.index[sel_df.index.values]
        sel_df.index = df.index
        df = df.loc[sel_index].copy()
        df[list(column_options)] = sel_df.loc[sel_index, list(column_options)]

    return df


def select_dataframe(df, column, label=None):
    label = None or f"select {column} to load"
    sel_val = st.selectbox(label, df[column])
    sel = df.loc[df[column] == sel_val].iloc[0]
    return sel


@st.cache_data
def df_to_csv(df, **kwargs):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(**kwargs).encode('utf-8')


def download_csv(df, name='data', button="Create download", key=None, **kwargs):
    with st.empty():
        if st.button(button, key=key):
            data = df_to_csv(df, **kwargs)
            st.download_button(
                label=f":inbox_tray: Download {name}.csv",
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


def set_plotly_inputs(
    data,
    x_cols=None,
    y_cols=None,
    color_cols=None,
    symbol_cols=None,
    facet_cols=None,
    facet_rows=None,
    category_orders=None,
    col_labels=None,
    max_unique=8,
    **kwargs
):
    cols = st.columns(6)
    numeric_columns = fields.filter_numerical_columns(data)
    cat_columns = fields.filter_categorical_columns(
        data, max_unique=max_unique)

    col_options = {
        "x": numeric_columns if x_cols is None else x_cols,
        "y": numeric_columns if y_cols is None else y_cols,
        "color": cat_columns if color_cols is None else color_cols,
        "symbol": cat_columns if symbol_cols is None else symbol_cols,
        "facet_col": np.r_[None, cat_columns] if facet_cols is None else facet_cols,
        "facet_row": np.r_[None, cat_columns] if facet_rows is None else facet_rows,
    }
    choose_cols = {k: len(opts) > 1 for k, opts in col_options.items()}
    col_labels = {
        "x": "Choose x",
        "y": "Choose y",
        "color": "Colour",
        "symbol": "Symbol",
        "facet_col": "Choose columns",
        "facet_row": "Choose rows"
    }
    col_labels.update(col_labels or {})
    n_cols = sum(choose_cols.values())
    cols = st.columns(n_cols)
    cat_orders = dict(
        data.apply(lambda s: sorted(s.unique())).items()
    )
    cat_orders.update(category_orders or {})
    plotly_inputs = {
        "data_frame": data,
        "category_orders": cat_orders,
    }

    i = 0
    for col, options in col_options.items():
        val = kwargs.get(col, options[0])
        if choose_cols[col]:
            with cols[i]:
                val = st.selectbox(
                    col_labels[col],
                    options=options,
                    index=list(options).index(val)
                )
            i += 1
        plotly_inputs[col] = val

    return plotly_inputs
