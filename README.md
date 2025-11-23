🧠 NeuroGuard – Real-Time EEG Epileptic Seizure Detection Web App

NeuroGuard is a Streamlit-based intelligent EEG analysis system powered by a CNN-LSTM deep learning model and a Flask API backend.
It provides real-time seizure detection, probability timelines, event morphology visualization, and adaptive EEG analysis through a clean and intuitive interface.

This project combines machine learning, signal processing, and modern web deployment into a powerful clinical-grade decision support tool.

🚀 Features
🔍 Real-Time Seizure Detection

Upload EEG signals (edf formats)

Automatic preprocessing and segmentation

CNN-LSTM model classifies seizure vs. non-seizure activity

Provides decision confidence in intuitive verbal format

📊 Interactive Visualizations

Probability timeline plot

Event morphology visualization

Adaptive filtered event representation

Multi-channel support (if enabled)

⚙️ Backend + Frontend Architecture

Streamlit UI for user interaction

Flask REST API for model inference

Modular design for easy deployment and scaling

🧩 Machine Learning Model

Hybrid CNN-LSTM architecture for temporal-spatial EEG feature extraction

Trained on epileptic EEG datasets

.h5 model file included for reproducibility

💾 Export Features

Auto-generated PDF reports (ReportLab)

Metadata tagging (timestamp, session info, etc.)

🧪 How It Works

User uploads an EEG signal file

Signal pre-processing

Filtering

Normalization

Windowing

CNN-LSTM model performs classification

CNN → extract spatial features

LSTM → capture temporal dependencies

Outputs:

Seizure probability

Verbal confidence interpretation

Event morphology graphs

Timeline evolution

Optional: Export results as PDF report
