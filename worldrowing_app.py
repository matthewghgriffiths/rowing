

import streamlit as st

st.set_page_config(
    page_title="Home",
)
st.title("World Rowing results visualisation app")

st.markdown("""
A [streamlit](https://streamlit.io/) webapp to load, process and analyse rowing data 
from [World Rowing](https://worldrowing.com/).

This app uses publicly available data from [World Rowing](https://worldrowing.com/) 
but is not endorsed or supported by World Rowing.

## Pages

### 1. [GMTs](/GMTs)
Allows loading, filtering and visualisation of results and PGMTs from a FISA competition.

### 2. [livetracker](/livetracker)
Allows loading, filtering and visualisation of livetracker data from a FISA competition.

The source code for this app can be found on 
[github.com/matthewghgriffiths/rowing](https://github.com/matthewghgriffiths/rowing)

License: [MIT](https://github.com/matthewghgriffiths/rowing/blob/main/LICENSE)
""")