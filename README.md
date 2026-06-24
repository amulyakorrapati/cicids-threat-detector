# Telecom Network Threat Detector

AI-powered network intrusion detection system built on the 
CICIDS 2017 dataset, inspired by research on AI in Cyber 
Security Risk Management for Telecom Industry 4.0.

## Features
- Detects DDoS attacks in real time using a Random Forest model
- UI dynamically built from the model's actual top 20 features
- Anti-overfitting measures applied (depth limiting, feature selection)
- Load real test samples to verify predictions instantly

## Dataset
CICIDS 2017 — Canadian Institute for Cybersecurity

## Tech Stack
- Python, Scikit-learn, Pandas, NumPy
- Gradio for the web interface
- Deployed on Hugging Face Spaces