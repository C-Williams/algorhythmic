import streamlit as st

# Imports for tensorflow
import tensorflow as tf
from tensorflow import keras

# Imports for scraping Spotify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from utils import (display_spotify, 
                   get_spotify_df,
                   predict_spotify,
                   name_dict,
                   get_spotify_recs)


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


@st.cache_resource
def load_scraping_model():
    # Set the model for Spotify Scraping
    scrape_model = keras.models.load_model('./models/keras_2/')
    return scrape_model

scrape_model = load_scraping_model()

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


st.markdown("# Let's test Spotify") 
st.write("")
st.write("")
st.write("### How to use.")
st.write("""
    This model uses the sound file present within Spotify's library to make predictions.
    For this reason, the predictions across both models for the same song will likely be
    different. Spotify's files have less distortion and are the full length of the song,
    therefore, they *may* have a more "accurate" result.
    
    To test, type in information about the song below and click *"Submit"*. You can give
    both the title of a song with or without an album name (if you type in the album, use 
    a comma to separate the two names).

    Double check that the album looks ilke what you expect, then press *"Yes"* or 
    *"No"* to see some insights about your song.
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