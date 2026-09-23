# 🎵 Music Mood Classification Using Machine Learning

An AI-powered web application that analyzes an uploaded music file and predicts its emotional characteristics using Machine Learning and audio signal processing.

The system extracts meaningful audio features such as MFCCs, Chroma, Spectral Centroid, Zero Crossing Rate, RMS Energy and Tempo, and uses them to estimate multiple emotional dimensions of music.

---

## 🎯 Objectives

- Analyze emotional characteristics present in music.
- Extract meaningful features from audio files.
- Apply Machine Learning techniques for emotion prediction.
- Predict multiple emotional dimensions from a single audio file.
- Provide an easy-to-use web interface for music analysis.
- Present prediction results through visual emotion scores.

---

## 🧠 Emotions Analyzed

The system analyzes the following nine emotional dimensions:

- Amazement
- Solemnity
- Tenderness
- Nostalgia
- Calmness
- Power
- Joyful Activation
- Tension
- Sadness

---

## 🎧 Audio Features

The system extracts several audio characteristics using Librosa:

| Feature | Description |
|---|---|
| MFCC | Represents the spectral characteristics of audio |
| Chroma | Represents pitch-class information |
| Spectral Centroid | Indicates the brightness of an audio signal |
| Zero Crossing Rate | Measures signal sign changes |
| RMS Energy | Represents signal energy |
| Tempo | Represents the estimated beats per minute |

These features are converted into numerical values and provided to the Machine Learning model.

---

## 🤖 Machine Learning

The project experiments with multiple Machine Learning approaches:

- Linear Regression
- Random Forest
- XGBoost

The task is treated as a **multi-output regression problem**, where the model predicts continuous scores for the nine emotional dimensions.

### Cross-Validation Results

The models were evaluated using 5-fold cross-validation.

| Model | Average MAE | Average MSE | Average R² |
|---|---:|---:|---:|
| Random Forest | 0.1247 | 0.0245 | 0.1888 |
| XGBoost | 0.1223 | 0.0245 | 0.1769 |

The current implementation uses the Random Forest model for the web application.

---

## 🔄 Project Workflow

```text
Audio File
    ↓
Audio Preprocessing
    ↓
Feature Extraction
    ↓
MFCC + Chroma + Spectral Features
    ↓
Machine Learning Model
    ↓
Multi-Output Emotion Prediction
    ↓
Emotion Scores
    ↓
Web Application