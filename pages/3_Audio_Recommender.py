import streamlit as st

import xgboost as xgb

from utils import (name_dict, 
                   predict_output,
                   get_spotify_recs)

# Imports for scraping Spotify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from audio_recorder_streamlit import audio_recorder


@st.cache_resource(ttl=3600)
def api_call():
    # Set up the Spotify client credentials
    client_id = st.secrets["CLIENT_ID"] # Also lives in .env
    client_secret = st.secrets["CLIENT_SECRET"] # Also lives in .env
    # Instantiate the response
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    return sp

sp = api_call()


@st.cache_resource
def load_listening_model():
    # Set the model for listening
    listen_model = xgb.XGBClassifier()
    listen_model.load_model('./models/new_xgb.h5')
    return listen_model

listen_model = load_listening_model()  
    

st.markdown("# Let's test some sounds!") 
st.write("")
st.write("")
st.write("### How to use.")
st.write("""
    As you are recording and testing, keep in mind that this site is being 
    hosted for free. Therefore, it may be a bit slow.
    
    To start, make sure you are ready with your sounds, then select the length 
    of time you wish to record and click the grey microphone. The microphone 
    should turn green to let you know that you are recording.

    You can then play back your sounds to see if the quality is sufficient,
    if not, click the microphone to record again. Once you are pleased with your
    recording, press *"Evaluate"* to see what genre your song may be and to
    see some recommendations from Spotify.
    
    *Some Notes:*
    - The listening model is only as good as the input, longer is typically 
    better, and less background noise is always appreciated!

    - In addition, because classical and hoerspiel music tends to be at a lower 
    average volume, if you get either of these and do not think they apply, try 
    playing the song louder.

    - You may hear that the audio quality is lacking, this is because of audio
    compression. Your speakers and microphone may not be the highest quality and
    thus neither is the recording.
    """)

slider = st.slider(label='Audio Length', min_value=3.0, max_value=30.0, step=1.0)

audio_bytes = audio_recorder(
    text="",
    recording_color="#5AC69F",
    neutral_color="#979797",
    icon_name="microphone",
    icon_size="6x",
    energy_threshold=(-1.0, 1.0),
    pause_threshold=slider,
)


if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")


if st.button("Evalute"):
    try:
        listen_preds = predict_output(listen_model, audio_bytes)
        st.write(listen_preds)

        st.write("Here are some less known songs from these genres!")
        get_spotify_recs(sp, listen_preds)

        st.write("""
        If you would like to test another sound, scroll up and click the 
        microphone again.""")
        
    except TypeError:
        st.write("""
        Be sure you have recorded some noise before pressing *"Evaluate"*.
        """)


st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.caption("Source code for recorder [here](https://github.com/Joooohan/audio-recorder-streamlit)")
st.caption("Built by Chris Williams as part of Spiced Academy Berlin")
