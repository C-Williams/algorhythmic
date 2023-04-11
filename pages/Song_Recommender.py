import streamlit as st

# Imports for tensorflow
import tensorflow as tf
from tensorflow import keras

import xgboost as xgb

from utils import (name_dict, 
                   display_spotify, 
                   get_spotify_df,
                   predict_spotify,
                   predict_output,
                   get_spotify_recs)

# Imports for scraping Spotify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from st_custom_components import st_audiorec

# Session states for Listening Prediction
if 'recorded' not in st.session_state:
    st.session_state.recorded = False

if 'evaluated' not in st.session_state:
    st.session_state.evaluated = False


# Session states for Spotify Prediction
if 'display' not in st.session_state:
    st.session_state.display = False

if 'shown_albums' not in st.session_state:
    st.session_state.shown_albums = False
    
if 'wrong_guess' not in st.session_state:
    st.session_state.wrong_guess = False

if "lock_form" not in st.session_state:
    st.session_state.lock_form = False

if "lock_buttons" not in st.session_state:
    st.session_state.lock_buttons = False

if "predicted_spotify" not in st.session_state:
    st.session_state.predicted_spotify = False

# Functions for predicting with Spotify
def on_yes():
    st.session_state.wrong_guess = False
    st.session_state.shown_albums = True
    st.session_state.lock_buttons = True
    st.session_state.lock_form = True

def on_no():
    st.session_state.display = False
    st.session_state.shown_albums = False
    st.session_state.wrong_guess = True
    st.session_state.lock_form = False

def lock_form():
    st.session_state.lock_form = True

def reset_form():
    st.session_state.lock_form = False
    st.session_state.display = False
    st.session_state.wrong_guess = False
    st.session_state.shown_albums = False
    st.session_state.lock_buttons = False
    st.session_state.predicted_spotify = False


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
def load_scraping_model():
    # Set the model for Spotify Scraping
    scrape_model = keras.models.load_model('./models/keras_2/')
    return scrape_model

scrape_model = load_scraping_model()


@st.cache_resource
def load_listening_model():
    # Set the model for listening
    listen_model = xgb.XGBClassifier()
    listen_model.load_model('./models/new_xgb.h5')
    return listen_model

listen_model = load_listening_model()  
    

st.markdown("# Let's test the models!") 
st.write("")
st.write("")
st.write("### There are actually two models here to test.")
st.write("")
st.write(
    """
    The first uses Spotify's API to pull out important features that
    are, in fact, found through the use of proprietary algorithms.
    Therefore, while we can scrape their API to pull out the features,
    we cannot, unfortunately, replicate them ourselves.
    
    The second model also makes use of Spotify's API to get some features,
    however, these fewer features are able to be replicated somewhat using
    the Librosa library.
    """)
st.write("### How to use.")
st.write("""
    Either fill out the form with your song's information, or simply skip the 
    form and hit *"Start Recording"* play at least 3 seconds of your sounds, 
    then press *"Stop"*.

    You can then play back your sounds to see if the quality is sufficient,
    if not, click *"Reset"* then record again. Once you are pleased with your
    recording, press *"Evaluate"* to see what genre your song may be and to
    see some recommendations from Spotify.
    
    *Some Notes:*
    - The listening model is only as good as the input, longer is typically 
    better, and less background noise is always appreciated!

    - In addition, because classical music tends to be at a lower average volumne, 
    if you get classical and do not think it applies, try playing the song louder.
    """)


@st.cache_data
def create_wav():
    with open('output.wav', mode='bw') as f:
        f.write(wav_audio_data)

wav_audio_data = st_audiorec()


if wav_audio_data is not None:
    st.session_state.recorded = True

if st.session_state.recorded:

    if st.button("Evalute"):
        try:
            create_wav()

            listen_preds = predict_output(model=listen_model)
            st.write(listen_preds)

            st.write("Here are some less known songs from these genres!")
            get_spotify_recs(sp, listen_preds)
            st.session_state.predicted_listen = True
            st.write("""
            If you would like to test another sound, scroll up and click *"Reset"*,
            then *"Start Recording"* again.""")
        except TypeError:
            st.write("""
            Be sure you have recorded some noise before pressing *"Evaluate"*.
            """)


with st.form("my_form"):
    st.write("Pick a song to play")
    song_title = st.text_input("What song are you curious about?")
    st.write("")
    artist_name = st.text_input("Who is it by?")
    st.write("")
    assigned_genre = st.selectbox(label="What genre would you give this song?", options=sorted(list(name_dict.values())))

   # Every form must have a submit button.
    submitted = st.form_submit_button("Submit", 
                                      on_click=lock_form,
                                      disabled=st.session_state.lock_form)

    if submitted:
        st.write("Check if we found the right song!")
        st.session_state.display = True

if st.session_state.display:
    
    display_spotify(sp, song_title, artist_name, 0)
    col1, col2 = st.columns(2)
    with col1:
        yes1 = st.button('Yes', 
                         key='yes1', 
                         on_click=on_yes,
                         disabled=st.session_state.lock_buttons)
    with col2:
        no1 = st.button('No', 
                        key='no1', 
                        on_click=on_no,
                        disabled=st.session_state.lock_buttons)

if st.session_state.wrong_guess:
    st.write('Maybe check the spelling?')

if st.session_state.shown_albums:
    try:
        df_spotify = get_spotify_df(sp, song_title, artist_name)


        st.write(df_spotify)

        st.write("")
        st.write("Reducing DataFrame and Making Predictions...")

        spotify_preds = predict_spotify(df_spotify, model=scrape_model)

        st.write(spotify_preds)

        if st.button("Would you like to some songs recommendations?", key='second'):
            get_spotify_recs(sp, spotify_preds)
    
        st.session_state.predicted_spotify = True
    
    except ValueError:
        st.write("""
        Unfortunately, it seems Spotify does not have audio analysis for this song.
        You will have to try a different song.""")

        st.button("Reset the search", on_click=reset_form)

if st.session_state.predicted_spotify:
    st.button("Would you like to test any other song?", on_click=reset_form, key='reset_button2')
    st.write("You know you want to!")


st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.caption("Built by Chris Williams as part of Spiced Academy Berlin")