import streamlit as st
import pandas as pd
import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go
import plotly.express as px

# Set the list of features that are important for the model
important_features = ['tempo','chroma0_mean','chroma1_mean',
                      'mfcc0_mean','mfcc1_mean','mfcc2_mean',
                      'mfcc3_mean', 'start_max_mean','max_loud_mean']

@st.cache_data
def make_full_df():
    dir_path = './data/'
    csv_files = [file for file in os.listdir(dir_path) if file.endswith('.csv')]
    df_list = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(dir_path, file))
        df_list.append(df)
    full_df = pd.concat(df_list, ignore_index=True)

    return full_df


@st.cache_data
def make_artist_df():
    artist_df = full_df.drop_duplicates(subset='title', keep='first')
    artist_df = artist_df[['artist','title','year','genre']]
    artist_df.reset_index(inplace=True, drop=True)

    return artist_df


@st.cache_data
def make_grouped_df():
    grouped_df = artist_df.groupby('genre')['title'].nunique().sort_values(ascending=False).reset_index()
    temp_year = artist_df.groupby('genre')[['year']].mean().round().astype(int)
    grouped_df = grouped_df.merge(temp_year, on='genre')
    grouped_df.columns = ['genre','count','avg_year']
    grouped_df['genre'] = grouped_df['genre'].str.title()

    return grouped_df


@st.cache_data
def make_year_df():
    year_df = artist_df.groupby('year')[['title']].count().reset_index()
    year_df = year_df[1:]
    temp_title = artist_df.groupby('year')['title'].unique().reset_index()
    temp_title = pd.DataFrame(temp_title)
    temp_title = temp_title[1:]
    year_df = year_df.merge(temp_title, on='year')
    year_df['year'] = year_df['year'].astype(str)
    year_df.columns = ['year','count','title']

    return year_df


@st.cache_data
def make_genre_plot():
    fig = px.bar(grouped_df, x='genre', y='count',
                hover_data=['count','avg_year'], color='genre',
                labels={'count':'Count of Songs','avg_year': 'Avg Release Year'}, height=600)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def make_year_plot():
    fig1 = px.bar(year_df, x='year', y='count',
                hover_data=['count'], color='year',
                labels={'title':'Count of Songs'}, height=600,
                color_discrete_sequence=px.colors.qualitative.G10)
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)


full_df = make_full_df()
artist_df = make_artist_df()
grouped_df = make_grouped_df()
year_df = make_year_df()


if "variable" not in st.session_state:
    st.session_state.variable = "tempo"

st.write("# Data Time!")
st.write("""
    Algorhythmic was trained on 4234 songs, by 4229 artists, grouped into 50 (mostly) 
    distinct genres.

    The artists with more than one song included all happen to be from the "Boy Band"
    genre and are:
    * 911
    * A1
    * All-4-One
    * Boyzone
    * O-Town
    """)

st.write("#### Artists and Songs")
st.dataframe(artist_df)

make_genre_plot()
make_year_plot()

st.write("#### Why these songs and genres?")

st.write("""
    As Algorhythmic was being developed, a big question that needed answering was which
    and how many genres should be chosen. In order to answer that question, the website
    [Every Noise at Once](https://everynoise.com/) was chosen.

    This website was built by Glenn McDonald who was the prinicpal engineer at a company
    called [The Echo Nest](https://en.wikipedia.org/wiki/The_Echo_Nest), a music intelligence
    and data platform, purchased by Spotify in 2014 for aruond 50 million Euros. After the
    purchase, Glenn built this incredible list and map which, in their words:
    
        "...is an ongoing attempt at an algorithmically-generated, 
        readability-adjusted scatter-plot of the musical genre-space, based on data 
        tracked and analyzed for 6,077 genre-shaped distinctions by Spotify as of 
        2023-04-02. The calibration is fuzzy, but in general down is more organic, 
        up is more mechanical and electric; left is denser and more atmospheric, 
        right is spikier and bouncier."
    """)
col1, col2 = st.columns(2)

with col1:
    st.write("This is a snippet of the first few items on the list")
    st.image("./songs_images/genre_list_en.png")

with col2:
    st.write("This is what the list for 'Dance Pop' looks like")
    st.image("./songs_images/dance_pop_en.png")

st.write("""
    Using these lists, I selected 50 genres in total. To determine which 50, I first
    ordered Every Noise at Once's full list by popularity (*Note: This is popularity in
    the USA*) and then appended each genre to the list, only if it was mostly unrelated
    to genres already appended to list.

    In other words, Pop, the first entry was added to the list, however, 'Dance Pop, 
    although it is second on the list, when we go into 'Dance Pop's' page, we can see
    the Every Noise at Once specifies that it is most closely related to 'Pop.' Therefore,
    'Dance Pop' and other entries like it were excluded.

    After this process, we are left with these 50 genres:
    """)

st.dataframe(artist_df['genre'].unique())

st.write("""#### How were the songs selected?""")
st.write("""
    Once the genres were selected, it was time to select songs. Luckily, Every Noise at Once 
    is run by employees at Spotify. Therefore, each genre on the list has a prebuilt
    playlist, full of songs that Spotify has pre-determined fit each genre. So, after some 
    webscraping, our list of songs is complete.
    """)

st.write("#### But how different are they really?")
st.write("""
    Each genre is quite closely related. The differences between each, for us, seem
    quite small. The best explanation for this is to simply show you. Interact with 
    the textbox below to change the graph in order to examine the different genres 
    based on each feature that was used in training Algorhythmic.

    For a more in depth analysis of each feature, see "What is Sound." However, in brief:

    * Tempo is how fast each song is
    * MFCC corresponds to a mathematical description for the 'timbre', or 'tone color'
        * MFCC 0 describes general loudness
        * MFCC 1 describes 'brightness'
        * Each successive MFCC gets more vague and difficult to define
    * Chroma describes which note is heard at a given time
        * Chroma 0 is C natural
        * Chroma 1 is C#
    * Start_Max and Max_Loud are proprietary terms that [Spotify](https://developer.spotify.com/documentation/web-api/reference/get-audio-analysis) has derived and describes as:
        * Combined, these components can be used to desctibe the "attack" of each segment of a song
    """)

variable = st.selectbox(label="Pick a value: ", options=important_features)

x = full_df[["genre", variable]]

f, ax = plt.subplots(figsize=(16, 10));
sns.boxplot(x = "genre", y = variable, data = x, palette = 'husl');

plt.title(f'{variable} Boxplot for Genres', fontsize = 25)
plt.xticks(fontsize = 14,rotation=90)
plt.yticks(fontsize = 10)
plt.xlabel("Genre", fontsize = 15)
plt.ylabel(variable, fontsize = 15)
st.pyplot(f)

st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.caption("Built by Chris Williams as part of Spiced Academy Berlin")