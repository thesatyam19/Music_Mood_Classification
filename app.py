from flask import Flask, render_template, request
import librosa
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

model = joblib.load("models/random_forest_emotion_model.pkl")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def extract_features(file_path):

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

    return pd.DataFrame([features])

@app.route("/", methods=["GET", "POST"])
def home():

    results = None

    if request.method == "POST":

        file = request.files["audio"]

        if file and file.filename:

            file_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(file_path)

            features = extract_features(file_path)

            features = features[model.feature_names_in_]

            predictions = model.predict(features)[0]

            results = sorted(
                zip(
                    [
                        "amazement",
                        "solemnity",
                        "tenderness",
                        "nostalgia",
                        "calmness",
                        "power",
                        "joyful_activation",
                        "tension",
                        "sadness"
                    ],
                    predictions
                ),
                key=lambda x: x[1],
                reverse=True
            )

    return render_template(
        "index.html",
        results=results
    )
if __name__ == "__main__":
    app.run(debug=True)