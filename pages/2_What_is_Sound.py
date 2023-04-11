import streamlit as st

st.write("""
    # Kind of a weird question, right?
    """)

st.text("")
st.text("")

st.write("""### What's in a sound?""")

st.write("""
    We hear sounds every day, but perhaps we don't think too much about them. But what
    actually is sound?

    According a presentation at the [University of Toronto](http://www.cs.toronto.edu/~gpenn/csc401/soundASR.pdf):

        "Sound is a pressure wave which is created by a vibrating object.This vibrations 
        set particles in the surrounding medium (typical air) in vibrational motion, 
        thus transporting energy through the medium.
        
        Since the particles are moving in parallel direction to the wave movement, the 
        sound wave is referred to as a longitudinal wave. The result of longitudinal 
        waves is the creation of compressions and rarefactions within the air."
    """)
st.image('./songs_images/airpressure.jpg')

st.text("")

st.write("""
    These compressions and rarefactions vibrate our eardrums and are interpreted by our brains
    as say, a bird, or fireworks, or a dog barking.

    In music, these sound waves get even more complicated.
    """)

st.text("")

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("https://i.gifer.com/YdBO.gif", caption = "From gifer.com")


st.text("")

st.write("""
    In addition to loudness, we have elements like rhythm, tempo, pitch, and harmony. 
    All of which are captured by these waves, but how can we pull out the individual
    features?
    """)
st.text("")
st.write("### Music Analysis Buzzwords!")

st.write("""
    * Fourier Transformation
    * Mel-Factor Cepstral Coefficients
    * Chromagrams""")

st.write("### Fourier and Fast Fourier Transformations")
st.image('./songs_images/fft.png', caption="FFT of David Bowie's Space Oddity")
st.write("""
    The Fourier transform is a mathematical technique that allows us to analyze a signal 
    in terms of its constituent frequencies. It takes a time-domain signal and converts it 
    into a frequency-domain representation, showing the amount of energy at each frequency 
    in the signal. This can be useful for tasks like filtering out unwanted frequencies, 
    compressing audio or image data, and understanding the properties of waves and 
    oscillations. 
    """)
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/FFT_of_Cosine_Summation_Function.svg/1024px-FFT_of_Cosine_Summation_Function.svg.png",
         caption="Credit to Wikipedia")
st.write("""
    In the above image you can see that the red wave is the signal as it moves through time.
    Each bump is the result of a louder moment in time.

    The blue, mountainous looking graph is the Fourier Transformed version of the same sounds.
    It shows how loud the red wave is at a given Hz level. In other words, how much of each note 
    is heard in the signal. In our example the notes at 10, 20, 30, 40, and 50 Hz are much louder.

    These images are the main way song recognition apps like [Shazam](https://www.shazam.com/home)
    work.
    """)
st.write("""
    * 10 Hz is very low, so low in fact that it sounds more like a woodpecker against a thick, 
    hollow tree. I could not replicate this sound.""")
st.write("""
    * 20 Hz sound like this (it's an out of tune low E on a piano)""")
st.audio('./songs_images/low_e.wav')
st.write("""
    * 30 Hz sounds like this (it's a *very* out of tune B natural (H for our European visitors))""")
st.audio('./songs_images/low_b.wav')
st.write("""
    * 40 Hz is exactly one octave higher than 20 Hz and sounds like this""")
st.audio('./songs_images/mid_e.wav')
st.write("""
    * 50 Hz sounds like this (another very out of tune note, somewhere between G and Ab)""")
st.audio('./songs_images/mid_ab.wav')

st.text("")
st.text("")

st.write("### Mel-Factor Coefficients")
st.write("""
    Here we are getting even more complicated in our mathematical explanations. In essence,
    the [Mel-Frequency Cepstral Coefficients](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/) 
    (MFCCs) are a complex representation of the characteristics of a given sound.

    It is commonly derived from the following:
    1. Take the Fourier transform of a signal.
    2. Map the powers of the spectrum obtained above onto the [Mel Scale](https://en.wikipedia.org/wiki/Mel_scale).
    3. Take the logs of the powers at each of the mel frequencies.
    4. Take the [discrete cosine transform](https://users.cs.cf.ac.uk/Dave.Marshall/Multimedia/node231.html) of the list of mel log powers, as if it were a signal.
    5. The MFCCs are the amplitudes of the resulting spectrum.
    """)
st.image("https://developer.spotify.com/assets/audio/Timbre_basis_functions.png",
         caption="From Spotify")
 
st.write("""
    According to [Spotify](https://developer.spotify.com/documentation/web-api/reference/get-audio-analysis):
        
        "It is a complex notion also referred to as sound color, texture, or tone 
        quality, and is derived from the shape of a segment's spectro-temporal surface, 
        independently of pitch and loudness. (The results of the analysis are highly 
        abstract, however, roughly speaking,) the first dimension represents the average 
        loudness of the segment; second emphasizes brightness; third is more closely 
        correlated to the flatness of a sound; fourth to sounds with a stronger attack; 
        etc."
    """)

st.text("")
st.text("")

st.write("### Chromagrams")
st.image("./songs_images/chroma.png", caption="Chromagram of Bowie's Space Oddity")
st.write("""
    [Chromagrams](https://towardsdatascience.com/learning-from-audio-pitch-and-chromagrams-5158028a505) 
    are graphical representations of how loud a given pitch is at a given time.
    They are calculated using the Fourier Transformation along with a specific filter. 
    The filter aims to project all the energy of the recorded sound into 12 bins. These 12
    bins correspond to the 12 pitches in Western music, irrespective of the octave a given
    note is in. By disregarding octave, we can then create a heat-map of how the pitch changes 
    over time.

    In the above graph, the more red a square is, the loud that pitch is at a given moment
    in time.
    """)

st.text("")
st.text("")

st.write("### Closing")
st.write("""
    Using the features explained above, we can, with the help of computers, do a 
    pretty good job of mathematically separating different genres from one another. In our
    case, we trained neural networks to classify more than 4 thousand songs into 50
    different genres.

    More explanation and visualizations of the data can be found on the EDA page.
    """)

col1, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("https://librosa.org/doc/latest/_static/librosa_logo_text.svg")

st.write("""
    Special thanks to the kind folks at [Librosa](https://librosa.org/doc/latest/index.html) 
    who have done amazing work in making a library that is easy to use and quite clear,
    especially given the complicated nature of the topic. They did all of the heavy lifting for us. 
    """)


st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.caption("Built by Chris Williams as part of Spiced Academy Berlin")
