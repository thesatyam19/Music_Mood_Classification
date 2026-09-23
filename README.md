# 🎵 Music Mood Classification Using Machine Learning

An AI-powered music emotion classification system that analyzes an uploaded MP3 file and predicts its emotional characteristics using machine learning.

## 📌 Project Overview

Music can express multiple emotions at the same time. This project uses audio signal processing and machine learning to analyze music and estimate nine different emotional dimensions from an audio file.

The system extracts meaningful audio features from the uploaded music and uses a trained Random Forest model to generate emotion scores.

## 🎯 Objectives

- Analyze music using audio signal processing
- Extract meaningful audio features
- Predict multiple emotional dimensions
- Display emotion scores in an easy-to-understand interface
- Build a practical machine learning application using Flask

## 🧠 Emotions Predicted

The system predicts scores for:

- Amazement
- Solemnity
- Tenderness
- Nostalgia
- Calmness
- Power
- Joyful Activation
- Tension
- Sadness

## ⚙️ Technologies Used

### Programming
- Python

### Machine Learning
- Scikit-learn
- Random Forest
- XGBoost
- Linear Regression

### Audio Processing
- Librosa
- NumPy

### Data Processing
- Pandas

### Web Application
- Flask
- HTML
- CSS
- JavaScript

## 🎵 Audio Features

The system extracts several features from the uploaded audio:

- Tempo
- MFCC
- Chroma Features
- Spectral Centroid
- Zero Crossing Rate
- RMS Energy

These features are used as inputs to the machine learning model.

## 🔄 Project Workflow

```text
MP3 Audio
    ↓
Audio Loading
    ↓
Feature Extraction
    ↓
MFCC + Chroma + Spectral Features
    ↓
Machine Learning Model
    ↓
Emotion Prediction
    ↓
Emotion Scores
    ↓
Web Interface