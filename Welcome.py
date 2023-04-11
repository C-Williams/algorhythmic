import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Algorhythmic! 👋")


st.markdown(
    """
    Algorhythmic combines two seemingly different concepts into one, amazing tool. 
    This is an app built specifically for Genre Classification and Song Recommendation
    through the thoughtful use of musical and machine learning knowledge.

    These models are not designed to completely, with 100% accuracy
    determine to which genre a certain song, or clip belongs. In fact,
    it is our opinion that this is, in fact, impossible. Because genres
    of music are so subjective, and each person can view a different song
    from a different point of view, it is impossible even for humans to
    say with 100% accuracy what genre a song is.
    """)

st.write("### Genre Classification")

st.markdown("""
    [What genre is this?](https://open.spotify.com/track/1bxEpNR75Hq3T2oF9AZjt8?si=e5b18a6626b24e01&nd=1) (Warning: Is explicit after 45 seconds)

    Music genre classification involves building a machine learning model that can 
    automatically classify songs into different genres based on their audio features 
    such as rhythm, tempo, melody, and harmony. The goal of this project is to develop 
    a system that can accurately recognize and categorize music into predefined genres,
    such as rock, hip hop, soul, and classical, without the need for human intervention.

    To create such a system, one needs to first collect a large dataset of music tracks 
    with genre labels, extract relevant audio features, and then use machine learning 
    algorithms to train a model on this data.

    Compared to image or text data, the field of machine learning for sound or audio data 
    is relatively underdeveloped. This is because sound is a complex and multidimensional 
    data type that is more difficult to represent and analyze compared to images or text.
    
    In this case of music, it is also highly subjective. What is rock? What is punk? These
    labels were created by humans and are therefore subject to human failings. 

    With that in mind, the goal of these models is to extract patterns.
    Whether that is how fast a song is, which notes are in the song, or
    *how* those notes sound, the models are designed to find these differences
    and compare them to other existing genres that it has seen before.
    """
)

st.write("### Want to learn more?")

st.markdown(
    """
    **👈 Select a link from the sidebar** to see some examples
    of what Algorhythmic can do!
    """
)

st.write("### Libraries Used")
st.image("./songs_images/libraries_used.png")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.caption("Built by Chris Williams as part of Spiced Academy Berlin")