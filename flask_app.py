"""
Music Sentiment Analysis - Flask Web Application
Professional web interface for music sentiment analysis
Author: Iman Humayun
"""

from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load models
print("🔄 Loading models...")
try:
    with open('multi_label_model.pkl', 'rb') as f:
        multi_label_model = pickle.load(f)
    with open('single_label_model.pkl', 'rb') as f:
        single_label_model = pickle.load(f)
    with open('mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    df_clean = pd.read_csv('music_data.csv')
    
    with open('features.txt', 'r') as f:
        available_features = f.read().strip().split(',')
    
    emotion_columns = mlb.classes_.tolist()
    print("✅ Models loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("ℹ️ Please run music_analyzer.py first to train models")
    multi_label_model = None
    single_label_model = None
    mlb = None
    df_clean = None
    available_features = []
    emotion_columns = []

# Prediction functions
def predict_music_sentiment(danceability, energy, loudness, speechiness,
                          acousticness, instrumentalness, liveness, tempo):
    """Predict music emotional tendency (single-label)"""
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

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "feature_contributions": {
            'danceability': danceability,
            'energy': energy,
            'loudness': loudness,
            'speechiness': speechiness,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'liveness': liveness,
            'tempo': tempo
        }
    }

def predict_music_emotions(danceability, energy, loudness, speechiness,
                          acousticness, instrumentalness, liveness, tempo):
    """Predict multiple emotion tags for music"""
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

    emotion_probs = {}
    for i, emotion in enumerate(emotion_columns):
        prob = probabilities[i][0][1] if len(probabilities[i][0]) > 1 else probabilities[i][0][0]
        emotion_probs[emotion] = float(prob)

    predicted_emotions = [emotion for i, emotion in enumerate(emotion_columns) if predictions[i] == 1]

    return {
        "predicted_emotions": predicted_emotions,
        "emotion_probabilities": emotion_probs,
        "feature_contributions": {
            'danceability': danceability,
            'energy': energy,
            'loudness': loudness,
            'speechiness': speechiness,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'liveness': liveness,
            'tempo': tempo
        }
    }

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎵 Music Sentiment Analyzer - Professional Edition</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
            animation: fadeInDown 0.8s ease;
        }

        .header h1 {
            font-size: 3em;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .main-card {
            background: white;
            border-radius: 25px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            animation: fadeInUp 0.8s ease;
        }

        .tabs {
            display: flex;
            gap: 10px;
          
