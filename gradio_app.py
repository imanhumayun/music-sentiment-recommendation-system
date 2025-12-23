"""
Music Sentiment Analysis - Gradio Interface
Simple and intuitive interface for music sentiment analysis
Author: Iman Humayun
"""

import gradio as gr
import pickle
import pandas as pd
import numpy as np

print("🔄 Loading models...")
try:
    with open('multi_label_model.pkl', 'rb') as f:
        multi_label_model = pickle.load(f)
    with open('single_label_model.pkl', 'rb') as f:
        single_label_model = pickle.load(f)
    with open('mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    
    emotion_columns = mlb.classes_.tolist()
    print("✅ Models loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("ℹ️ Please run music_analyzer.py first to train models")
    exit(1)

def analyze_music_single(danceability, energy, loudness, speechiness,
                        acousticness, instrumentalness, liveness, tempo):
    """Analyze music sentiment (single-label)"""
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
    
    feature_info = "\n".join([
        f"• Danceability: {danceability:.2f}",
        f"• Energy: {energy:.2f}",
        f"• Loudness: {loudness:.2f} dB",
        f"• Speechiness: {speechiness:.2f}",
        f"• Acousticness: {acousticness:.2f}",
        f"• Instrumentalness: {instrumentalness:.2f}",
        f"• Liveness: {liveness:.2f}",
        f"• Tempo: {tempo:.2f} BPM"
    ])
    
    return (
        f"🎵 {prediction}",
        f"{confidence:.2%}",
        feature_info
    )

def analyze_music_multi(danceability, energy, loudness, speechiness,
                       acousticness, instrumentalness, liveness, tempo):
    """Analyze music emotions (multi-label)"""
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
    
    predicted_emotions = [emotion for i, emotion in enumerate(emotion_columns) if predictions[i] == 1]
    emotions_str = "🎨 " + ", ".join(predicted_emotions) if predicted_emotions else "No emotions detected"
    
    emotion_probs = {}
    for i, emotion in enumerate(emotion_columns):
        prob = probabilities[i][0][1] if len(probabilities[i][0]) > 1 else probabilities[i][0][0]
        emotion_probs[emotion] = prob
    
    probs_str = "\n".join([f"• {emotion}: {prob:.2%}" for emotion, prob in sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)])
    
    return (emotions_str, probs_str)

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="🎵 Music Sentiment Analyzer") as demo:
    gr.Markdown("# 🎵 Music Sentiment Analyzer")
    gr.Markdown("**AI-powered music emotion analysis using Random Forest models**")
    
    with gr.Tabs():
        # Single-Label Analysis Tab
        with gr.TabItem("🎯 Single-Label Analysis"):
            gr.Markdown("### Classify music into Positive, Negative, or Neutral sentiment")
            
            with gr.Row():
                with gr.Column():
                    g_dance = gr.Slider(0, 1, 0.5, step=0.01, label="🕺 Danceability")
                    g_energy = gr.Slider(0, 1, 0.5, step=0.01, label="⚡ Energy")
                    g_loud = gr.Slider(-60, 0, -10, step=1, label="🔊 Loudness (dB)")
                    g_speech = gr.Slider(0, 1, 0.1, step=0.01, label="🗣️ Speechiness")
                
                with gr.Column():
                    g_acoustic = gr.Slider(0, 1, 0.5, step=0.01, label="🎸 Acousticness")
                    g_inst = gr.Slider(0, 1, 0.0, step=0.01, label="🎹 Instrumentalness")
                    g_live = gr.Slider(0, 1, 0.1, step=0.01, label="🎤 Liveness")
                    g_tempo = gr.Slider(0, 250, 120, step=1, label="🥁 Tempo (BPM)")
            
            g_analyze = gr.Button("🔍 Analyze Sentiment", variant="primary")
            
            with gr.Row():
                g_pred = gr.Textbox(label="Predicted Sentiment", scale=1)
                g_conf = gr.Textbox(label="Confidence", scale=1)
            
            g_features = gr.Textbox(label="Audio Features", lines=8)
            
            g_analyze.click(
                analyze_music_single,
                inputs=[g_dance, g_energy, g_loud, g_speech, g_acoustic, g_inst, g_live, g_tempo],
                outputs=[g_pred, g_conf, g_features]
            )
        
        # Multi-Label Analysis Tab
        with gr.TabItem("🎨 Multi-Label Analysis"):
            gr.Markdown("### Detect multiple emotions: Happy, Sad, Energetic, Calm, etc.")
            
            with gr.Row():
                with gr.Column():
                    m_dance = gr.Slider(0, 1, 0.5, step=0.01, label="🕺 Danceability")
                    m_energy = gr.Slider(0, 1, 0.5, step=0.01, label="⚡ Energy")
                    m_loud = gr.Slider(-60, 0, -10, step=1, label="🔊 Loudness (dB)")
                    m_speech = gr.Slider(0, 1, 0.1, step=0.01, label="🗣️ Speechiness")
                
                with gr.Column():
                    m_acoustic = gr.Slider(0, 1, 0.5, step=0.01, label="🎸 Acousticness")
                    m_inst = gr.Slider(0, 1, 0.0, step=0.01, label="🎹 Instrumentalness")
                    m_live = gr.Slider(0, 1, 0.1, step=0.01, label="🎤 Liveness")
                    m_tempo = gr.Slider(0, 250, 120, step=1, label="🥁 Tempo (BPM)")
            
            m_analyze = gr.Button("🔍 Analyze Emotions", variant="primary")
            
            m_emotions = gr.Textbox(label="Predicted Emotions", scale=2)
            m_probs = gr.Textbox(label="Emotion Probabilities", lines=10)
            
            m_analyze.click(
                analyze_music_multi,
                inputs=[m_dance, m_energy, m_loud, m_speech, m_acoustic, m_inst, m_live, m_tempo],
                outputs=[m_emotions, m_probs]
            )
        
        # About Tab
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## 🎵 Music Sentiment Analyzer
            
            ### 🎯 Features
            - **Single-Label Classification**: Positive, Negative, or Neutral
            - **Multi-Label Emotion Detection**: Happy, Sad, Energetic, Calm, Excited, Melancholy, Relaxed, Aggressive
            - **Audio Feature Analysis**: 8 key Spotify audio features
            
            ### 🧠 Machine Learning
            - **Algorithm**: Random Forest Classifier
            - **Dataset**: Spotify Music Dataset (Kaggle)
            - **Accuracy**: ~80% for both single and multi-label models
            
            ### 🎼 Audio Features
            1. **Danceability** - How suitable the track is for dancing
            2. **Energy** - Intensity and activity level
            3. **Loudness** - Overall volume in decibels
            4. **Speechiness** - Presence of spoken words
            5. **Acousticness** - Likelihood of being acoustic
            6. **Instrumentalness** - Amount of instrumental content
            7. **Liveness** - Presence of audience
            8. **Tempo** - Speed in beats per minute
            
            ### 👨‍💻 Author
            **Iman Humayun** - AI/ML Engineer
            
            ### 📚 Tech Stack
            - Python 3.8+
            - scikit-learn
            - Gradio
            - Pandas
            - NumPy
            """)

print("\n" + "="*70)
print("🎵 GRADIO INTERFACE")
print("="*70)
print("\n🚀 Launching Gradio application...")
print("📍 The interface will open in your browser automatically")
print("\n💡 Press Ctrl+C to stop the application")
print("="*70 + "\n")

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
