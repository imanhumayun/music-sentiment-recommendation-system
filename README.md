Music Sentiment, Emotion Analyzer & Recommendation System
This project predicts music sentiment (single-label) and multiple emotions (multi-label) from Spotify style audio features and provides three user interfaces: Streamlit, Flask and Gradio. It also includes a simple song recommendation feature using similarity (nearest neighbors).
Features

* Single-label sentiment prediction: Positive / Neutral / Negative
* Multi-label emotion prediction: multiple emotions for the same track (e.g., Happy + Energetic)
* Recommendation system: finds similar songs using audio-feature similarity
* Three frontends included:

Streamlit dashboard
Flask web app (UI + JSON endpoints)
Gradio interface

Repository Contents


music_analyzer.py
Main training script. Loads data, creates labels, trains models, and saves artifacts needed by the apps.

streamlit_app.py
Streamlit dashboard for interactive predictions and analytics.

flask_app.py
Flask web application with a built-in HTML UI and prediction endpoints.

gradio_app.py
Gradio interface for quick testing of sentiment and emotion prediction.

music_sentiment_recommendation_system - Copy.ipynb
Notebook version of the full project.

requirements.txt
Python dependencies.

Setup
Create an environment (optional but recommended)
   python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
2) Install dependencies
pip install -r requirements.txt
3) Train Models (Run First)
Run the training script to generate the saved models and dataset used by the apps:
python music_analyzer.py

After training, you should have files created locally such as:

* single_label_model.pkl
* multi_label_model.pkl
* mlb.pkl
* music_data.csv
* features.txt

Run the Applications

Option A: Streamlit
streamlit run streamlit_app.py
Option B: Flask
python flask_app.py
Then open the local address shown in your terminal (typically http://localhost:5000).
Flask API endpoints

* POST /predict → single-label sentiment prediction
* POST /predict_multi → multi-label emotion prediction

Option C: Gradio
python gradio_app.py
Then open the local address shown in your terminal (typically http://127.0.0.1:7860).

Input Features Used


The models use these audio features:

* danceability
* energy
* loudness
* speechiness
* acousticness
* instrumentalness
* liveness
* tempo

Machine Learning Models
Single-Label Sentiment Classification

* Algorithm: Random Forest Classifier
* Classes: Positive, Negative, Neutral
* Accuracy: ~80%
* Use Case: Overall mood classification

Multi-Label Emotion Detection

* Algorithm: Multi-output Random Forest
* Emotions: Happy, Sad, Energetic, Calm, Excited, Melancholy, Relaxed, Aggressive
* Accuracy: ~80%
* Use Case: Detecting multiple emotions simultaneously

Recommendation System

* Algorithm: K-Nearest Neighbors (KNN)
* Metric: Euclidean distance
* Use Case: Finding similar songs based on audio features

Technologies Used
Core Technologies

* Python 3.8+ - Programming language
* scikit-learn - Machine learning library
* pandas - Data manipulation
* NumPy - Numerical computing

Web Frameworks

* Streamlit - Interactive dashboards
* Flask - Web application framework
* Gradio - ML interface builder

Data & Visualization

* Kaggle API - Dataset download
* Matplotlib - Static visualizations
* Seaborn - Statistical visualizations
* Plotly - Interactive charts

Dataset
Source: Spotify Music Dataset on Kaggle
Size: 1000+ songs
Features: 12+ audio features
Format: CSV
The dataset is automatically downloaded when you run music_analyzer.py for the first time.

Use Cases

* Music Streaming Platforms: Automatic playlist generation based on mood
* Mental Health Apps: Music therapy recommendations
* DJ Software: Mood-based track selection
* Music Production: Emotion-aware composition tools
* Research: Music psychology and emotion studies

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.

Author
GitHub: @imanhumayun
