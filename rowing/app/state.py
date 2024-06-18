
import json
import logging
import datetime
import copy

import streamlit as st

import numpy as np
from pandas.api.types import is_datetime64_any_dtype

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if is_datetime64_any_dtype(obj):
            return str(obj)
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return str(obj)

        return super(NumpyEncoder, self).default(obj)


def as_json(val):
    try:
        return json.dumps(val, cls=NumpyEncoder)
    except TypeError:
        return None


STATE = {
    k: v[0]
    for k, v in st.query_params.items()
}

get = STATE.get
items = STATE.items
keys = STATE.keys
values = STATE.values
clear = STATE.clear


def set(key, val):
    STATE[key] = val
    update_query_params()


def update(*args, **kwargs):
    STATE.update(*args, **kwargs)
    update_query_params()


def update_query_params():
    STATE.update(st.session_state)
    items = ((k, as_json(v)) for k, v in STATE.items())
    update_params = {
        k: v for k, v in items if v is not None
    }
    st.query_params.update(**update_params)


def get_state():
    return copy.deepcopy(STATE)


def reset_button(label='reset'):
    if st.button(label):
        st.session_state.clear()
        st.cache_resource.clear()
        clear()
        st.rerun()
