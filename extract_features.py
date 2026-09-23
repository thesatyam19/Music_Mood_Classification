import os
import librosa
import numpy as np
import pandas as pd

audio_folders = [
    "classical",
    "rock",
    "electronic",
    "pop"
]

genre_offset = {
    "classical": 0,
    "rock": 100,
    "electronic": 200,
    "pop": 300
}

audio_data = []

for folder in audio_folders:

    folder_path = os.path.join(".", folder)

    for file in os.listdir(folder_path):

        if not file.lower().endswith(".mp3"):
            continue

        file_path = os.path.join(folder_path, file)

        try:
            file_number = int(os.path.splitext(file)[0])
            track_id = file_number + genre_offset[folder]

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
                "track_id": track_id,
                "genre": folder,
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

            audio_data.append(features)

            print("Processed:", file_path, "→ Track ID:", track_id)

        except Exception as e:
            print("Error:", file_path)
            print(e)

audio_df = pd.DataFrame(audio_data)

print("\nAudio files processed:", len(audio_df))
print("Unique track IDs:", audio_df["track_id"].nunique())

print("\nTrack ID range:")
print(audio_df["track_id"].min(), "to", audio_df["track_id"].max())

os.makedirs("features", exist_ok=True)

audio_df.to_csv(
    "features/audio_features.csv",
    index=False
)

print("\nSaved: features/audio_features.csv")