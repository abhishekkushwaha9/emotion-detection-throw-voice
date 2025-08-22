import sounddevice as sd
import soundfile as sf
import joblib
from features import extract_features
import numpy as np

# Record from mic
def record_voice(seconds=4, sr=16000, filename="input.wav"):
    print("🎤 Recording started...")
    audio = sd.rec(int(seconds*sr), samplerate=sr, channels=1)
    sd.wait()
    sf.write(filename, audio, sr)
    print("✅ Recording saved:", filename)
    return filename

# Predict function
def predict_emotion(file_path, model_path="models/emotion_svm.joblib"):
    model = joblib.load(model_path)
    feat = extract_features(file_path).reshape(1, -1)
    return model.predict(feat)[0]

if __name__ == "__main__":
    file = record_voice(4)   # 4 sec record
    emotion = predict_emotion(file)
    print("👉 Detected Emotion:", emotion)
