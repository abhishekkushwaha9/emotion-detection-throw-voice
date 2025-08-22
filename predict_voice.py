import sounddevice as sd
import soundfile as sf
import numpy as np
import joblib
from features import extract_features

# Load pipeline (scaler + SVM inside)
model = joblib.load("models/emotion_svm.joblib")

def record_voice(duration=4, fs=22050, filename="input_voice.wav"):
    print("🎤 Speak now...")
    rec = sd.rec(int(duration*fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    sf.write(filename, rec, fs)
    print(f"✅ Recording saved as {filename}")
    return filename

def predict_emotion(file_path):
    feat = extract_features(file_path).reshape(1, -1)
    # scaler is inside the pipeline, so just predict:
    return model.predict(feat)[0]

if __name__ == "__main__":
    wav = record_voice(4)
    emo = predict_emotion(wav)
    print("🧠 Detected Emotion:", emo)
