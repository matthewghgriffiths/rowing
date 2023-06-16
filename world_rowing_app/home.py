

import streamlit as st

import sys 
import os
from pathlib import Path 

DIRPATH = Path(__file__).resolve().parent
LIBPATH = str(DIRPATH.parent)
realpaths = [os.path.realpath(p) for p in sys.path]
if LIBPATH not in realpaths:
    sys.path.append(LIBPATH)

st.set_page_config(
    page_title="Home",
)
st.title("World Rowing results visualisation app")

st.markdown("""
A [streamlit](https://streamlit.io/) webapp to load, process and analyse rowing data 
from [World Rowing](https://worldrowing.com/).

This app uses publicly available data from World Rowing 
but is not endorsed or supported by World Rowing.

## Pages

### 1. [GMTs](/GMTs)
Allows loading, filtering and visualisation of results and PGMTs from a FISA competition.

### 2. [livetracker](/livetracker)
Allows loading, filtering and visualisation of livetracker data from a FISA competition.

The livetracker data does not come with any time information, 
except for the intermediate times, so the app estimates the race time 
for each time for each timepoint to match as closely as possible the intermediate
and final times. 

From these estimated times the app can calculate distance from PGMT, 
which is the distance behind (or ahead) from a boat going at an even 
percentage gold medal pace. The percentage of this pace can be set in the app,
and defaults to 100%

### 3. [realtime](/realtime)
Allows the folling of the livetracker data from a race in realtime.

The source code for this app can be found on 
[github.com/matthewghgriffiths/rowing](https://github.com/matthewghgriffiths/rowing)

License: [MIT](https://github.com/matthewghgriffiths/rowing/blob/main/LICENSE)
""")
