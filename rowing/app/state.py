
import json 
import atexit
import logging

import streamlit as st 

import numpy as np

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
        return super(NumpyEncoder, self).default(obj)


STATE = {
    k: json.loads(v[0]) 
    for k, v in  st.experimental_get_query_params().items()
}

get = STATE.get 
set = STATE.__setitem__
items = STATE.items 
keys = STATE.keys 
values = STATE.values 
update = STATE.update 

def __setitem__(key, val):
    STATE[key] = val

def update_query_params():
    STATE.update(st.session_state)
    update_params = {
        k: json.dumps(v, cls=NumpyEncoder) for k, v in STATE.items()
    }
    st.experimental_set_query_params(**update_params)


atexit.register(update_query_params)