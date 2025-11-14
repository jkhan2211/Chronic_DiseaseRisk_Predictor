import streamlit as st
import requests
import time
import os
import socket

# =============================
# Symptom list
# =============================
SYMPTOM_LIST = [
    "itching", "skin_rash", "nodal_skin_eruptions",
    "continuous_sneezing", "shivering", "chills",
    "joint_pain", "stomach_pain", "acidity",
    "ulcers_on_tongue", "vomiting", "fatigue",
    "weight_gain", "anxiety", "headache",
    "back_pain", "nausea", "loss_of_appetite"
]


# =============================
# UI Layout
# =============================
col1, col2 = st.columns(2)

# -----------------------------
# LEFT SIDE – SYMPTOM PICKER
# -----------------------------
with col1:
    st.markdown("### 🩺 Select Symptoms (Multi-Select)")

    selected_symptoms = st.multiselect(
        "Choose all symptoms you are experiencing:",
        SYMPTOM_LIST
    )

    # Convert to model-ready dictionary
    symptoms = {
        symptom: 1 if symptom in selected_symptoms else 0
        for symptom in SYMPTOM_LIST
    }

    if st.button("🔍 Predict Disease", use_container_width=True):
        with st.spinner("Analyzing symptoms..."):
            try:
                api_endpoint = os.getenv("API_URL", "http://localhost:8000")
                predict_url = f"{api_endpoint.rstrip('/')}/predict"

                payload = {"symptoms": symptoms}

                response = requests.post(predict_url, json=payload)
                response.raise_for_status()

                st.session_state.prediction = response.json()
                st.session_state.prediction_time = time.time()

            except Exception as e:
                st.error(f"Error contacting API: {e}")
                st.stop()


# -----------------------------
# RIGHT SIDE – SHOW SELECTION + RESULT
# -----------------------------
with col2:
    st.markdown("### 👍 Selected Symptoms")

    if selected_symptoms:
        st.success(f"Selected ({len(selected_symptoms)}):")
        st.write(selected_symptoms)
    else:
        st.info("No symptoms selected.")

    # Show model output
    if "prediction" in st.session_state:
        result = st.session_state.prediction

        st.markdown("---")
        st.markdown("### 🧬 Predicted Disease")
        st.success(result["predicted_disease"])

        st.markdown("### 🔝 Top 3 Predictions")
        for item in result["top_3"]:
            st.write(f"**{item['disease']}** — {round(item['probability']*100,1)}%")

        st.markdown(f"⏱ Prediction Time: `{result['prediction_time']}`")
