# CodeAlpha Emotion Recognition from Speech (Task 2)

An end-to-end machine learning pipeline built to extract audio features and classify emotions from speech using modern Python libraries. Developed as part of the CodeAlpha Machine Learning Internship.

## 🚀 Project Overview

This project processes audio signals to accurately recognize and classify human emotions (such as happy, sad, angry, neutral). It extracts robust audio characteristics using `librosa` and trains a multi-layer perceptron neural network classifier.

* **Architecture:** Multi-Layer Perceptron (MLP) Classifier / Neural Network
* **Libraries:** `librosa`, `scikit-learn`, `numpy`
* **Environment:** Google Colab / Local Python 3.x

## 📊 Key Components & Workflow

1. **Feature Extraction (`librosa`):** Extracts robust audio descriptors:
   * **MFCCs (Mel-Frequency Cepstral Coefficients):** Captures the overall spectral envelope.
   * **Chroma Frequencies:** Represents tonal and harmonic characteristics.
   * **Mel Spectrogram:** Measures short-term power spectrum across Mel scale frequencies.
2. **Model Training (`scikit-learn`):** Trains an optimized `MLPClassifier` with fully configured activation layers and solver parameters for high classification accuracy.
3. **Warning-Free Code:** Built with modern, deprecation-free syntax and explicit data typing for optimal performance.

## 🛠️ Repository Structure

* `CodeAlpha_EmotionRecognition.ipynb`: Interactive Jupyter Notebook containing the full execution workflow.
* `CodeAlpha_EmotionRecognition.py`: Clean, standalone Python script for the feature extraction pipeline.
* `README.md`: Project documentation.
