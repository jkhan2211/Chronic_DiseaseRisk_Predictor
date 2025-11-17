# Disease Feature Classifier – Streamlit App
A Streamlit web application for predicting possible diseases based on selected symptoms. The app connects to a backend API for predictions and can be run locally or in Docker.

## Features
    - Multi-select symptoms picker
    - Predicts disease based on selected symptoms
    - Displays top 3 predicted diseases with probabilities
    - Shows prediction timestamp
    - Simple two-column layout: left for input, right for results

## Folder Contents
    - main.py -> Streamlit app script
    - Dockerfile -> Docker setup
    - requirements.txt -> Python dependencies

## Quick Start
*Using Docker*
    ```
    docker build -t disease-classifier .
    docker run -p 8501:8501 disease-classifier
    ```
    Open http://localhost:8501 in your browser

*Without Docker*
    ```
    pip install -r requirements.txt
    streamlit run main.py
    ```

## Notes
The app calls a backend API (/predict). Make sure the API is running.
Default Streamlit port: 8501.

## License
MIT License