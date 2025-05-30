

import streamlit as st

from rowing.world_rowing import pages


def main():
    st.title("World Rowing results visualisation app")
    st.markdown(
        """
        A [streamlit](https://streamlit.io/) webapp to load, process and analyse rowing data 
        from [World Rowing](https://worldrowing.com/).

        This app uses publicly available data from World Rowing 
        but is not endorsed or supported by World Rowing.

        ## Pages

        ### [1. GMTs](/GMTs)
        Allows loading, filtering and visualisation of results and PGMTs from a FISA competition.

        ### [2. Livetracker](/livetracker)
        Allows loading, filtering and visualisation of livetracker data from a FISA competition.

        The livetracker data does not come with any time information, 
        except for the intermediate times, so the app estimates the race time 
        for each time for each timepoint to match as closely as possible the intermediate
        and final times. 

        From these estimated times the app can calculate distance from PGMT, 
        which is the distance behind (or ahead) from a boat going at an even 
        percentage gold medal pace. The percentage of this pace can be set in the app,
        and defaults to 100%

        ### [3. Realtime](/realtime)
        Allows the folling of the livetracker data from a race in realtime.

        ### [4. Results](/results)
        Shows a summary of the results of a competition

        ### [5. Entries](/entries)
        Shows a summary of the entries to a competition
                        

        ## Notes

        If data isn't loading correctly you may need to clear the cache,
        which you can do in the settings.

        The source code for this app can be found on
        [github.com/matthewghgriffiths/rowing](https://github.com/matthewghgriffiths/rowing)
        
        ## QR Code
        """
    )
    st.image(
        "https://qrcode.tec-it.com/API/QRCode"
        "?data=https%3A%2F%2Fworldrowing.streamlit.app"
        # "&backcolor=%23ffffff"
        "&size=small"
        "&quietzone=1&errorcorrection=H"
    )


if __name__ == "__main__":
    st.set_page_config(page_title="World Rowing", layout='wide')
    pg = st.navigation([
        st.Page(main, title='World Rowing'),
        st.Page(pages.pgmts.main, title='PGMTs', url_path='pgmts'),
        st.Page(pages.livetracker.main, title='Livetracker',
                url_path='livetracker'),
        st.Page(pages.realtime.main, title='Realtime', url_path='realtime'),
        st.Page(pages.results.main, title='Results', url_path='results'),
        st.Page(pages.entries.main, title='Entries', url_path='entries'),
    ])
    pg.run()
