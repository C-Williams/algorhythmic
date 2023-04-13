# Import the standards
import pandas as pd
import numpy as np

# Imports for scraping Spotify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Imports for Librosa
import librosa

# Import for reading in bytes type file
import soundfile as sf
import io

# Imports for tensorflow
import tensorflow as tf
from tensorflow import keras

import xgboost as xgb

from sklearn import preprocessing

import streamlit as st
import os

# Set the list of features that are important for the model
important_features = ['tempo','start_max_std','mfcc3_mean','start_max_mean','loud_time_mean',
                      'max_loud_mean','chroma1_mean','mfcc10_mean','mfcc5_mean','mfcc3_std',
                      'mfcc7_mean','max_loud_std','mfcc1_mean','chroma0_mean','mfcc9_mean',
                      'mfcc2_mean','mfcc0_mean']

# Set the list of possible genres
name_dict = {0: 'pop',1: 'rap',2: 'modern rock',3: 'urbano latino',4: 'edm',5: 'latin pop',
            6: 'classic rock',7: 'r&b',8: 'musica mexicana',9: 'alternative metal',
            10:'contemporary country',11: 'k-pop',12: 'canadian pop',13: 'filmi',14: 'indie pop',
            15: 'folk rock',16: 'neo mellow',17: 'french hip hop',18: 'adult standards',
            19: 'arrocha',20: 'new wave pop',21: 'german hip hop',22: 'house',23: 'j-pop',
            24: 'turkish pop',25: 'soul',26: 'metal',27: 'indonesian pop',28: 'conscious hip hop',
            29: 'stomp and holler',30: 'italian hip hop',31: 'pop punk',32: 'disco',33: 'hollywood',
            34: 'singer-songwriter',35: 'trap argentino',36: 'dark trap',37: 'hoerspiel',38: 'indie soul',
            39: 'nu jazz',40: 'boy band',41: 'desi hip hop',42: 'electronica',43: 'permanent wave',
            44: 'indietronica',45: 'punk',46: 'modern blues',47: 'vapor trap',48: 'mpb',49: 'classical'}



# Set up the scaling process
def scale_data(df, scaler = preprocessing.MinMaxScaler()):
    cols = df.columns
    np_scaled = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(np_scaled, columns = cols)
    return scaled_df


# Set up a function to condense long songs into smaller chunks
def condense_spotify_data(df, duration, fade_in, fade_out):

    # Find the number of rows that equal 3 seconds
    i = round(len(df) / duration * 3)

    # Accounting for songs with very few segments
    if i < 2:
        mean = df.groupby(np.arange(len(df)) // 2).mean()
        std = df.groupby(np.arange(len(df)) // 2).std()

    else:
        # Find the number of rows to drop such that the resulting DataFrame divides evenly 
        # into i. Note: all dataframes will always be the same length.
        drop_point = len(df) % i
        # First check which is longer, fade in/out, then drop the longer one from the DataFrame
        # This ensures that dropped info is less important.
        if fade_in > fade_out:
            mean = df[drop_point:].groupby(np.arange(len(df[drop_point:])) // i).mean()
            std = df[drop_point:].groupby(np.arange(len(df[drop_point:])) // i).std()
        else:
            mean = df[:-drop_point].groupby(np.arange(len(df[:-drop_point])) // i).mean()
            std  = df[:-drop_point].groupby(np.arange(len(df[:-drop_point])) // i).std()
    
    to_return = mean.merge(std, left_index=True, right_index=True)

    return to_return

def condense_output_data(df, duration):

    # Find the number of rows that equal 3 seconds
    i = round(len(df) / duration * 3)

    # Accounting for songs with very few segments
    if i < 2:
        mean = df.groupby(np.arange(len(df)) // 2).mean()
        std = df.groupby(np.arange(len(df)) // 2).std()

    else:
        # Find the number of rows to drop such that the resulting DataFrame divides evenly 
        # into i. Note: all dataframes will always be the same length.
        drop_point = len(df) % i
       
        mean = df[drop_point:].groupby(np.arange(len(df[drop_point:])) // i).mean()
        std = df[drop_point:].groupby(np.arange(len(df[drop_point:])) // i).std()
    
    to_return = mean.merge(std, left_index=True, right_index=True)

    return to_return


def display_spotify(sp, song_title, artist_name, num):
    
    # Search for the track
    results = sp.search(q=f"{song_title}, {artist_name}", type='track')
    
    items = results['tracks']['items']
    
    if len(items) == 0:
        st.write("Double check your spelling...")
    else:
        for item in items[num:num+1]:

            image_url = item['album']['images'][1]['url']
            album_song = item['name']
            album_name = item['album']['name']
            album_artist = item['album']['artists'][0]['name']

            st.write(f"Is {album_song} by {album_artist}, off the album {album_name}, correct?")
            st.image(image_url)


def get_spotify_df(sp, song_title, artist_name):
    st.write("Creating DataFrame...")
    st.write("For a deeper explanation of these values see 'What is Sound'")

    temp_df = pd.DataFrame()

    results = sp.search(q=f"{song_title}, {artist_name}", type='track')
    track = results['tracks']['items'][0]

    track_title = track['name']
    artist_name = track['artists'][0]['name']
    release_date = track['album']['release_date']
    audio_analysis = sp.audio_analysis(track['id'])
    track_info = audio_analysis.get('track')
    duration = track_info.get('duration')

    fade_in = track_info.get('end_of_fade_in')
    fade_out = duration - track_info.get('start_of_fade_out') 

    tempo = round(track_info.get('tempo'))

    pitch_list = list()
    timbre_list = list()
    loud_start_list = list()
    loud_max_time_list = list()
    loud_max_list = list()

    for segment in audio_analysis['segments']:

        loud_start = segment.get('loudness_start')
        loud_max_time = segment.get('loudness_max_time')
        loud_max = segment.get('loudness_max')
        pitch_array = segment.get('pitches')
        timbre_array = segment.get('timbre')

        loud_start_list.append(loud_start)
        loud_max_time_list.append(loud_max_time)
        loud_max_list.append(loud_max)
        pitch_list.append(pitch_array)
        timbre_list.append(timbre_array)

    loud_start_df = pd.DataFrame(loud_start_list).astype('float16')
    loud_start_df = scale_data(loud_start_df)
    loud_start_df = condense_spotify_data(loud_start_df, duration, fade_in, fade_out)
    loud_start_df.columns = ['start_max_mean', 'start_max_std']
    loud_start_df[loud_start_df.columns] = loud_start_df.astype('float16')

    loud_max_time_df = pd.DataFrame(loud_max_time_list).astype('float16')
    loud_max_time_df = condense_spotify_data(loud_max_time_df, duration, fade_in, fade_out)
    loud_max_time_df.columns = ['loud_time_mean', 'col2']
    loud_max_time_df.drop(['col2'],axis=1,inplace=True )
    loud_max_time_df[loud_max_time_df.columns] = loud_max_time_df.astype('float16')
    loud_max_time_df['title'] = track_title
    loud_max_time_df['artist'] = artist_name
    loud_max_time_df['year'] = release_date[:4]
    loud_max_time_df['tempo'] = tempo
    loud_max_time_df['tempo'] = loud_max_time_df['tempo'].astype('uint8')

    loud_max_df = pd.DataFrame(loud_max_list).astype('float16')
    loud_max_df = scale_data(loud_max_df)
    loud_max_df = condense_spotify_data(loud_max_df, duration, fade_in, fade_out)
    loud_max_df.columns = ['max_loud_mean','max_loud_std']
    loud_max_df[loud_max_df.columns] = loud_max_df.astype('float16')

    timbre_df = pd.DataFrame(timbre_list)
    timbre_df = scale_data(timbre_df)
    timbre_df = condense_spotify_data(timbre_df, duration, fade_in, fade_out)
    timbre_df.columns = ['mfcc0_mean','mfcc1_mean','mfcc2_mean','mfcc3_mean','mfcc4_mean','mfcc5_mean','mfcc6_mean',
                        'mfcc7_mean','mfcc8_mean','mfcc9_mean','mfcc10_mean','mfcc11_mean','mfcc0_std','mfcc1_std',
                        'mfcc2_std','mfcc3_std','mfcc4_std','mfcc5_std','mfcc6_std','mfcc7_std','mfcc8_std','mfcc9_std',
                        'mfcc10_std','mfcc11_std']
    timbre_df[timbre_df.columns] = timbre_df.astype('float16')

    pitch_df = pd.DataFrame(pitch_list)
    pitch_df = condense_spotify_data(pitch_df, duration, fade_in, fade_out)
    pitch_df.columns = ['chroma0_mean','chroma1_mean','chroma2_mean','chroma3_mean','chroma4_mean',
                        'chroma5_mean','chroma6_mean','chroma7_mean','chroma8_mean','chroma9_mean','chroma10_mean',
                        'chroma11_mean','chroma0_std','chroma1_std','chroma2_std','chroma3_std','chroma4_std','chroma5_std',
                        'chroma6_std','chroma7_std','chroma8_std','chroma9_std','chroma10_std','chroma11_std']
    pitch_df[pitch_df.columns] = pitch_df[pitch_df.columns].astype('float16')

    combined_song_df = pd.concat([timbre_df,
                                pitch_df,
                                loud_max_df,
                                loud_start_df,
                                loud_max_time_df
                                ],
                                axis = 1)

    temp_df = pd.concat([temp_df, combined_song_df])

    return(temp_df)


def predict_spotify(df, model):
    
    df = df[important_features]
        
    df_keras = tf.convert_to_tensor(df)
    
    probs_list = []
    preds = model.predict(df_keras)
    for pred in preds:
        probs_list.append(np.argsort(pred)[::-1][:3])

    probs_df = pd.DataFrame(probs_list).replace(name_dict)
    probs_df.columns = ['First Guess','Second Guess','Third Guess']

    counts_one = probs_df['First Guess'].value_counts() * 3
    counts_two = probs_df['Second Guess'].value_counts() * 2
    counts_three = probs_df['Third Guess'].value_counts()

    merge_test = pd.concat([counts_one, counts_two, counts_three], axis=1)
    merge_test['Weighted Votes'] = merge_test.sum(axis=1)
    merge_test = merge_test.sort_values(by='Weighted Votes',ascending=False)
    new_df = pd.DataFrame(merge_test.head(5))
    st.write("This is our guess at what the genre is:")

    return new_df


def predict_output(model, audio_bytes):

    y, sr = librosa.load(io.BytesIO(audio_bytes))
    duration = librosa.get_duration(y=y, sr=sr)
    audio_file, _ = librosa.effects.trim(y)
    hop_length = 512

    mfccs = librosa.feature.mfcc(y=audio_file, sr=sr, n_mfcc=12, hop_length=hop_length)
    mfcc_df = pd.DataFrame(mfccs.T)
    scaled_mfcc = scale_data(mfcc_df)
    condensed_mfcc= condense_output_data(scaled_mfcc, duration)
    condensed_mfcc.columns = ['mfcc0_mean','mfcc1_mean','mfcc2_mean','mfcc3_mean','mfcc4_mean',
                              'mfcc5_mean','mfcc6_mean','mfcc7_mean','mfcc8_mean','mfcc9_mean',
                              'mfcc10_mean','mfcc11_mean','mfcc0_std','mfcc1_std','mfcc2_std',
                              'mfcc3_std','mfcc4_std','mfcc5_std','mfcc6_std','mfcc7_std',
                              'mfcc8_std','mfcc9_std','mfcc10_std','mfcc11_std']
    
    chromagram = librosa.feature.chroma_stft(y=audio_file, sr=sr, hop_length=hop_length)
    chromagram_df = pd.DataFrame(chromagram.T)
    scaled_chroma = scale_data(chromagram_df)
    condensed_chroma = condense_output_data(scaled_chroma, duration)
    condensed_chroma.columns = ['chroma0_mean','chroma1_mean','chroma2_mean','chroma3_mean',
                                'chroma4_mean','chroma5_mean','chroma6_mean','chroma7_mean',
                                'chroma8_mean','chroma9_mean','chroma10_mean','chroma11_mean',
                                'chroma0_std','chroma1_std','chroma2_std','chroma3_std',
                                'chroma4_std','chroma5_std','chroma6_std','chroma7_std',
                                'chroma8_std','chroma9_std','chroma10_std','chroma11_std']
    
    tempo, beats = librosa.beat.beat_track(y=audio_file, sr=sr)
    tempo = round(tempo)

    combined = pd.concat([condensed_mfcc, condensed_chroma], axis=1)
    combined['tempo'] = tempo

    probs_list = []
    pred_probs = model.predict_proba(combined)

    for pred in pred_probs:
        probs_list.append(np.argsort(pred)[::-1][:3])

    probs_df = pd.DataFrame(probs_list).replace(name_dict)
    probs_df.columns = ['First Guess','Second Guess','Third Guess']

    counts_one = probs_df['First Guess'].value_counts() * 3
    counts_two = probs_df['Second Guess'].value_counts() * 2
    counts_three = probs_df['Third Guess'].value_counts()

    merge_test = pd.concat([counts_one, counts_two, counts_three], axis=1)
    merge_test = merge_test.fillna(0)
    merge_test['Weighted Votes'] = merge_test.sum(axis=1)
    merge_test = merge_test.sort_values(by='Weighted Votes',ascending=False)
    new_df = pd.DataFrame(merge_test.head(5))
    
    return new_df


def get_spotify_recs(sp, df):

    genre_list = df.index.to_list()
    
    artists = []
    tracks = []
    urls = []
    images = []

    for genre in genre_list:
        
        results = sp.search(q=f"genre:{genre}, tag:hipster", type='track', limit=50, market='US')
        
        if genre =='pop':
            results = sp.search(q="genre:pop, tag:hipster", type='track',limit=50, market='US')

        choice = np.random.choice(len(results['tracks']['items']))

        track = results['tracks']['items'][choice]['name']
        artist = results['tracks']['items'][choice]['artists'][0]['name']
        url = results['tracks']['items'][choice]['preview_url']
        image = results['tracks']['items'][choice]['album']['images'][1]['url']
        tracks.append(track)
        artists.append(artist)
        urls.append(url)
        images.append(image)
    
    for artist, track, url, image in zip(artists, tracks, urls, images):

        st.write(f"Try: {track} by {artist}")
        st.image(image)
        st.audio(url)
    