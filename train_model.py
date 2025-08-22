import os
import glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# 📂 Dataset path
data_path = "dataset"

# ✅ Get all .wav files (including subfolders)
wav_files = glob.glob(os.path.join(data_path, "**", "*.wav"), recursive=True)

if not wav_files:
    print("❌ No .wav files found! Check dataset folder path.")
    exit()

print(f"📂 Dataset Path: {data_path}")
print(f"🎵 Total audio files found: {len(wav_files)}")

# 🎯 Function to extract features
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

# 🎯 Features and labels
X, y = [], []

for file in wav_files:
    features = extract_features(file)
    if features is not None:
        X.append(features)

        # Label extraction (example: RAVDESS filenames -> "03-01-01-01-01-01-01.wav")
        # 3rd part (index=2) usually denotes emotion
        label = int(file.split("-")[2])  
        y.append(label)

X = np.array(X)
y = np.array(y)

print("✅ Features Extracted:", X.shape, y.shape)

# ❌ Check if data is empty
if len(X) == 0 or len(y) == 0:
    print("❌ No features extracted. Please check dataset and filenames.")
    exit()

# 🎯 Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Train model (SVM)
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# 🎯 Test accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# 💾 Save model
joblib.dump(model, "emotion_model.pkl")
print("💾 Model saved as emotion_model.pkl")
