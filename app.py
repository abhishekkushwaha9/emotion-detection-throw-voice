import streamlit as st
import librosa
import numpy as np
import pickle

# ✅ Load trained model
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Emotion map
emotion_map = {
    "neutral": "😐 Neutral",
    "calm": "😌 Calm",
    "happy": "😃 Happy",
    "sad": "😢 Sad",
    "angry": "😠 Angry",
    "fearful": "😨 Fearful",
    "disgust": "🤢 Disgust",
    "surprised": "😲 Surprised"
}

st.title("🎵 Speech Emotion Detection")
st.write("Upload an audio file (.wav) and I will predict the emotion!")

# ✅ File uploader
uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:
    # Load audio
    signal, sr = librosa.load(uploaded_file, sr=None)
    
    # Extract features
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T, axis=0)
    mfccs = mfccs.reshape(1, -1)

    # Predict emotion
    prediction = model.predict(mfccs)[0]
    
    st.subheader("🎯 Predicted Emotion:")
    st.success(emotion_map.get(prediction, prediction))

    # Play audio
    st.audio(uploaded_file, format="audio/wav")
