"""
Music Sentiment Analysis - Streamlit Dashboard
Interactive web application for analyzing music sentiment and emotions
Author: Iman Humayun
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title="🎵 Music Sentiment Analyzer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    h1, h2, h3 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    try:
        with open('multi_label_model.pkl', 'rb') as f:
            multi_model = pickle.load(f)
        with open('single_label_model.pkl', 'rb') as f:
            single_model = pickle.load(f)
        with open('mlb.pkl', 'rb') as f:
            mlb_obj = pickle.load(f)
        df = pd.read_csv('music_data.csv')
        return multi_model, single_model, mlb_obj, df
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("ℹ️ Please run music_analyzer.py first to train and save models.")
        return None, None, None, None

multi_label_model, single_label_model, mlb, df_clean = load_models()

feature_cols = ['danceability', 'energy', 'loudness', 'speechiness',
               'acousticness', 'instrumentalness', 'liveness', 'tempo']

# Sidebar
with st.sidebar:
    st.markdown("# 🎵 Navigation")
    page = st.radio("Choose a page:", [
        "🏠 Home",
        "🎯 Single-Label Analysis",
        "🎨 Multi-Label Analysis",
        "🎼 Song Recommendations",
        "📊 Dataset Analytics"
    ])

    st.markdown("---")
    st.markdown("### 📈 Model Info")
    if df_clean is not None:
        st.metric("Training Samples", len(df_clean))
        st.metric("Features", len(feature_cols))
        if mlb is not None:
            st.metric("Emotion Labels", len(mlb.classes_))

# Home Page
if page == "🏠 Home":
    st.markdown("<h1 style='text-align: center;'>🎵 Music Sentiment Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white; font-size: 1.2em;'>Analyze music emotions using AI and machine learning</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; color: white;'>
        <h3>🎯 Single-Label</h3>
        <p>Classify music into Positive, Negative, or Neutral sentiment</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; color: white;'>
        <h3>🎨 Multi-Label</h3>
        <p>Detect multiple emotions like Happy, Sad, Energetic, Calm</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; color: white;'>
        <h3>🎼 Recommendations</h3>
        <p>Find similar songs based on audio features</p>
        </div>
        """, unsafe_allow_html=True)

# Single-Label Analysis
elif page == "🎯 Single-Label Analysis":
    st.markdown("<h1>🎯 Single-Label Sentiment Analysis</h1>", unsafe_allow_html=True)

    if single_label_model is None:
        st.error("❌ Model not loaded. Please train models first.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        danceability = st.slider("🕺 Danceability", 0.0, 1.0, 0.5, 0.01)
        energy = st.slider("⚡ Energy", 0.0, 1.0, 0.5, 0.01)
        loudness = st.slider("🔊 Loudness (dB)", -60.0, 0.0, -10.0, 1.0)
        speechiness = st.slider("🗣️ Speechiness", 0.0, 1.0, 0.1, 0.01)

    with col2:
        acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.5, 0.01)
        instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.0, 0.01)
        liveness = st.slider("🎤 Liveness", 0.0, 1.0, 0.1, 0.01)
        tempo = st.slider("🥁 Tempo (BPM)", 0.0, 250.0, 120.0, 1.0)

    if st.button("🔍 Analyze Sentiment", key="single"):
        input_features = pd.DataFrame([{
            'danceability': danceability,
            'energy': energy,
            'loudness': loudness,
            'speechiness': speechiness,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'liveness': liveness,
            'tempo': tempo
        }])

        prediction = single_label_model.predict(input_features)[0]
        confidence = max(single_label_model.predict_proba(input_features)[0])

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 30px; border-radius: 15px; color: white; text-align: center;'>
        <h2>🎵 Predicted Sentiment: {prediction}</h2>
        <h3>Confidence: {confidence:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': "Confidence"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "darkgreen"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

# Multi-Label Analysis
elif page == "🎨 Multi-Label Analysis":
    st.markdown("<h1>🎨 Multi-Label Emotion Analysis</h1>", unsafe_allow_html=True)

    if multi_label_model is None:
        st.error("❌ Model not loaded. Please train models first.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        danceability = st.slider("🕺 Danceability", 0.0, 1.0, 0.5, 0.01, key="m_dance")
        energy = st.slider("⚡ Energy", 0.0, 1.0, 0.5, 0.01, key="m_energy")
        loudness = st.slider("🔊 Loudness (dB)", -60.0, 0.0, -10.0, 1.0, key="m_loud")
        speechiness = st.slider("🗣️ Speechiness", 0.0, 1.0, 0.1, 0.01, key="m_speech")

    with col2:
        acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.5, 0.01, key="m_acoustic")
        instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.0, 0.01, key="m_inst")
        liveness = st.slider("🎤 Liveness", 0.0, 1.0, 0.1, 0.01, key="m_live")
        tempo = st.slider("🥁 Tempo (BPM)", 0.0, 250.0, 120.0, 1.0, key="m_tempo")

    if st.button("🔍 Analyze Emotions", key="multi"):
        input_features = pd.DataFrame([{
            'danceability': danceability,
            'energy': energy,
            'loudness': loudness,
            'speechiness': speechiness,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'liveness': liveness,
            'tempo': tempo
        }])

        predictions = multi_label_model.predict(input_features)[0]
        probabilities = multi_label_model.predict_proba(input_features)

        predicted_emotions = [emotion for i, emotion in enumerate(mlb.classes_) if predictions[i] == 1]

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        padding: 30px; border-radius: 15px; color: white; text-align: center;'>
        <h2>🎨 Predicted Emotions</h2>
        <h3>{", ".join(predicted_emotions)}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Emotion probabilities
        emotion_probs = {}
        for i, emotion in enumerate(mlb.classes_):
            prob = probabilities[i][0][1] if len(probabilities[i][0]) > 1 else probabilities[i][0][0]
            emotion_probs[emotion] = prob

        fig = px.bar(
            x=list(emotion_probs.keys()),
            y=list(emotion_probs.values()),
            labels={'x': 'Emotion', 'y': 'Probability'},
            title='Emotion Probabilities'
        )
        st.plotly_chart(fig, use_container_width=True)

# Song Recommendations
elif page == "🎼 Song Recommendations":
    st.markdown("<h1>🎼 Song Recommendation System</h1>", unsafe_allow_html=True)

    if df_clean is None:
        st.error("❌ Data not loaded. Please train models first.")
        st.stop()

    song_options = []
    for i, row in df_clean.iterrows():
        track = str(row.get('track_name', f'Track {i}'))[:50]
        artist = str(row.get('artist_name', f'Artist {i}'))[:30]
        song_options.append(f"{i}: {track} by {artist}")

    selected_song = st.selectbox("🎵 Select a Song", song_options[:100])
    num_recs = st.slider("Number of Recommendations", 1, 10, 5)

    if st.button("🔍 Get Recommendations"):
        try:
            song_index = int(selected_song.split(":")[0])

            song_features = df_clean.iloc[song_index][feature_cols].to_frame().T
            knn = NearestNeighbors(n_neighbors=num_recs+1, metric='euclidean')
            knn.fit(df_clean[feature_cols])
            distances, indices = knn.kneighbors(song_features)

            st.markdown("### 🎵 Selected Song")
            selected_info = df_clean.iloc[song_index]
            st.write(f"**Track:** {selected_info.get('track_name', 'Unknown')}")
            st.write(f"**Artist:** {selected_info.get('artist_name', 'Unknown')}")

            st.markdown("### 🎼 Recommendations")
            for idx in indices[0][1:]:
                rec = df_clean.iloc[idx]
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.1); padding: 15px;
                border-radius: 10px; margin: 10px 0; color: white;'>
                <h4>{rec.get('track_name', 'Unknown')}</h4>
                <p>Artist: {rec.get('artist_name', 'Unknown')}</p>
                <p>Sentiment: {rec.get('sentiment', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Dataset Analytics
elif page == "📊 Dataset Analytics":
    st.markdown("<h1>📊 Dataset Analytics</h1>", unsafe_allow_html=True)

    if df_clean is None:
        st.error("❌ Data not loaded. Please train models first.")
        st.stop()

    st.markdown("### Dataset Overview")
    st.write(f"Total Songs: {len(df_clean)}")

    # Feature distributions
    st.markdown("### Feature Distributions")
    feature = st.selectbox("Select Feature", feature_cols)
    fig = px.histogram(df_clean, x=feature, title=f'{feature.capitalize()} Distribution')
    st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix
    st.markdown("### Feature Correlations")
    corr = df_clean[feature_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
