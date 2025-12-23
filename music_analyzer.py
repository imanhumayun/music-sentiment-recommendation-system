"""
Music Sentiment Analysis - Model Training Script
Trains Random Forest models for sentiment classification and emotion detection
Author: Iman Humayun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
import os
import kagglehub
import pickle

print("="*70)
print("🎵 MUSIC SENTIMENT ANALYSIS - MODEL TRAINING")
print("="*70)

# Step 1: Load Dataset from Kaggle
print("\n📥 Step 1: Loading dataset from Kaggle...")
try:
    path = kagglehub.dataset_download("solomonameh/spotify-music-dataset")
    print("✅ Path to dataset files:", path)
    
    files = os.listdir(path)
    print(f"📁 Files in dataset directory: {files}")
    
    csv_files = [f for f in files if f.endswith('.csv')]
    if csv_files:
        csv_path = os.path.join(path, csv_files[0])
        print(f"📂 Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✅ Dataset loaded successfully with {len(df)} rows")
        print(f"📊 Available columns: {df.columns.tolist()}")
    else:
        raise FileNotFoundError("No CSV file found")
        
except Exception as e:
    print(f"⚠️ Kaggle download failed: {e}")
    print("🔄 Creating sample dataset...")
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'danceability': np.random.uniform(0, 1, n_samples),
        'energy': np.random.uniform(0, 1, n_samples),
        'loudness': np.random.uniform(-20, 0, n_samples),
        'speechiness': np.random.uniform(0, 1, n_samples),
        'acousticness': np.random.uniform(0, 1, n_samples),
        'instrumentalness': np.random.uniform(0, 1, n_samples),
        'liveness': np.random.uniform(0, 1, n_samples),
        'tempo': np.random.uniform(60, 200, n_samples),
        'valence': np.random.uniform(0, 1, n_samples),
        'track_name': [f'Track {i}' for i in range(n_samples)],
        'artist_name': [f'Artist {i%100}' for i in range(n_samples)]
    })
    print(f"✅ Sample dataset created with {len(df)} rows")

# Step 2: Data Preprocessing and Multi-label Emotion Creation
print("\n🏷️ Step 2: Creating emotion labels...")

def create_emotion_tags(valence, energy):
    """Create multiple emotion tags based on valence and energy values"""
    tags = []
    
    if valence >= 0.7:
        tags.append("Happy")
    elif valence <= 0.3:
        tags.append("Sad")
    else:
        tags.append("Neutral")
    
    if energy >= 0.7:
        tags.append("Energetic")
    elif energy <= 0.3:
        tags.append("Calm")
    
    if valence >= 0.6 and energy >= 0.6:
        tags.append("Excited")
    elif valence <= 0.4 and energy <= 0.4:
        tags.append("Melancholy")
    elif valence >= 0.6 and energy <= 0.4:
        tags.append("Relaxed")
    elif valence <= 0.4 and energy >= 0.6:
        tags.append("Aggressive")
    
    return tags

# Apply emotion tags
if 'valence' in df.columns and 'energy' in df.columns:
    df['emotion_tags'] = df.apply(lambda row: create_emotion_tags(row['valence'], row['energy']), axis=1)
else:
    all_tags = ["Happy", "Sad", "Neutral", "Energetic", "Calm", "Excited", "Melancholy", "Relaxed", "Aggressive"]
    df['emotion_tags'] = [np.random.choice(all_tags, size=np.random.randint(1, 4), replace=False).tolist()
                          for _ in range(len(df))]

# Create binary columns for each emotion tag
mlb = MultiLabelBinarizer()
emotion_dummies = pd.DataFrame(mlb.fit_transform(df['emotion_tags']),
                              columns=mlb.classes_,
                              index=df.index)
df = pd.concat([df, emotion_dummies], axis=1)

# Select feature columns
feature_cols = ['danceability', 'energy', 'loudness', 'speechiness',
               'acousticness', 'instrumentalness', 'liveness', 'tempo']

available_features = [col for col in feature_cols if col in df.columns]
emotion_columns = mlb.classes_.tolist()

print(f"✅ Using features: {available_features}")
print(f"✅ Emotion labels: {emotion_columns}")

# Remove missing values
df_clean = df.dropna(subset=available_features + emotion_columns)
df_clean = df_clean.reset_index(drop=True)
print(f"✅ Clean dataset size: {len(df_clean)}")

# Step 3: Exploratory Data Analysis
print("\n📊 Step 3: Creating visualizations...")

plt.figure(figsize=(12, 8))
emotion_counts = df_clean[emotion_columns].sum().sort_values(ascending=False)
sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
plt.title('Emotion Tag Distribution in Music Dataset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('emotion_distribution.png')
print("✅ Saved: emotion_distribution.png")
plt.close()

plt.figure(figsize=(12, 8))
sns.heatmap(df_clean[available_features].corr(), annot=True, cmap='coolwarm')
plt.title('Audio Features Correlation Matrix')
plt.tight_layout()
plt.savefig('feature_correlation.png')
print("✅ Saved: feature_correlation.png")
plt.close()

# Step 4: Train Multi-Label Model
print("\n🧠 Step 4: Training multi-label emotion model...")
X = df_clean[available_features]
y = df_clean[emotion_columns]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

multi_label_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
multi_label_model.fit(X_train, y_train)

y_pred = multi_label_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Multi-label Model Accuracy: {accuracy:.4f}")

# Step 5: Train Single-Label Model
print("\n🧠 Step 5: Training single-label sentiment model...")
df_clean['sentiment'] = df_clean['valence'].apply(lambda v: "Positive" if v >= 0.6 else ("Negative" if v <= 0.4 else "Neutral"))
y_single = df_clean['sentiment']

X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
    X, y_single, test_size=0.2, random_state=42, stratify=y_single
)

single_label_model = RandomForestClassifier(n_estimators=100, random_state=42)
single_label_model.fit(X_train_single, y_train_single)

y_pred_single = single_label_model.predict(X_test_single)
accuracy = accuracy_score(y_test_single, y_pred_single)
print(f"✅ Single-label Model Accuracy: {accuracy:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': single_label_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📈 Feature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance for Music Sentiment Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("✅ Saved: feature_importance.png")
plt.close()

# Step 6: Save Models and Data
print("\n💾 Step 6: Saving models and data...")

with open('multi_label_model.pkl', 'wb') as f:
    pickle.dump(multi_label_model, f)
print("✅ Saved: multi_label_model.pkl")

with open('single_label_model.pkl', 'wb') as f:
    pickle.dump(single_label_model, f)
print("✅ Saved: single_label_model.pkl")

with open('mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)
print("✅ Saved: mlb.pkl")

df_clean.to_csv('music_data.csv', index=False)
print("✅ Saved: music_data.csv")

# Save feature list
with open('features.txt', 'w') as f:
    f.write(','.join(available_features))
print("✅ Saved: features.txt")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print("\n📦 Generated files:")
print("  • multi_label_model.pkl (Multi-label emotion classifier)")
print("  • single_label_model.pkl (Single-label sentiment classifier)")
print("  • mlb.pkl (Multi-label binarizer)")
print("  • music_data.csv (Processed dataset)")
print("  • features.txt (Feature list)")
print("  • emotion_distribution.png")
print("  • feature_correlation.png")
print("  • feature_importance.png")
print("\n🚀 You can now run the web applications:")
print("  • python streamlit_app.py")
print("  • python flask_app.py")
print("  • python gradio_app.py")
