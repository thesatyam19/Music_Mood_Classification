import librosa
import numpy as np
import pandas as pd
import joblib

file_path = input("Enter audio file path: ")

y, sr = librosa.load(file_path, sr=None)

mfcc = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=13
)

chroma = librosa.feature.chroma_stft(
    y=y,
    sr=sr
)

spectral_centroid = librosa.feature.spectral_centroid(
    y=y,
    sr=sr
)

zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

rms = librosa.feature.rms(y=y)

tempo = librosa.feature.tempo(
    y=y,
    sr=sr
)[0]

features = {
    "tempo": tempo
}

for i in range(13):
    features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
    features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

for i in range(12):
    features[f"chroma_{i+1}_mean"] = np.mean(chroma[i])
    features[f"chroma_{i+1}_std"] = np.std(chroma[i])

features["spectral_centroid_mean"] = np.mean(spectral_centroid)
features["spectral_centroid_std"] = np.std(spectral_centroid)

features["zero_crossing_rate_mean"] = np.mean(zero_crossing_rate)
features["zero_crossing_rate_std"] = np.std(zero_crossing_rate)

features["rms_mean"] = np.mean(rms)
features["rms_std"] = np.std(rms)

X = pd.DataFrame([features])

model = joblib.load(
    "models/random_forest_emotion_model.pkl"
)

predictions = model.predict(X)[0]

emotions = [
    "amazement",
    "solemnity",
    "tenderness",
    "nostalgia",
    "calmness",
    "power",
    "joyful_activation",
    "tension",
    "sadness"
]

result = pd.DataFrame({
    "Emotion": emotions,
    "Score": predictions
})

result = result.sort_values(
    "Score",
    ascending=False
)

print("\nPREDICTED EMOTIONS")
print("=" * 40)
print(result.to_string(index=False))