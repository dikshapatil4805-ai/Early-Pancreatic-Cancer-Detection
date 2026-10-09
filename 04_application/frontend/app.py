import json

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Pancreatic Cancer Research",
    page_icon="🔬",
    layout="wide",
)

st.title("Early Pancreatic Cancer Detection")
st.subheader("Explainable Multimodal Deep Learning")

st.warning(
    "Research prototype only. This application is not a medical "
    "diagnostic tool and must not be used to make treatment decisions."
)

try:
    response = requests.get(f"{API_URL}/health", timeout=3)
    response.raise_for_status()
    st.success("Backend API is connected.")
except requests.RequestException:
    st.error(
        "Backend is not running. Start the FastAPI server and refresh this page."
    )
    st.stop()

st.markdown(
    """
    Upload a CT scan and enter the clinical information required by the
    research prototype. Predictions will remain unavailable until the
    trained models and multimodal fusion pipeline are configured.
    """
)

with st.form("prediction_form"):
    st.subheader("1. CT scan")

    ct_file = st.file_uploader(
        "Select a CT file",
        type=["dcm", "nii", "gz", "zip"],
        help="Supported extensions: .dcm, .nii, .nii.gz, .zip",
    )

    st.subheader("2. Clinical information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=120,
            value=50,
            step=1,
        )

        sex = st.selectbox(
            "Sex",
            ["Not specified", "Female", "Male", "Other"],
        )

    with col2:
        family_history = st.selectbox(
            "Family history",
            ["Not specified", "Yes", "No"],
        )

        smoking_history = st.selectbox(
            "Smoking history",
            ["Not specified", "Yes", "No"],
        )

    symptoms_duration_days = st.number_input(
        "Symptom duration in days (if applicable)",
        min_value=0,
        max_value=36500,
        value=0,
        step=1,
    )

    submitted = st.form_submit_button("Submit for research processing")

if submitted:
    if ct_file is None:
        st.error("Please select a CT file.")
        st.stop()

    clinical_data = {
        "age": int(age),
        "sex": None if sex == "Not specified" else sex,
        "family_history": (
            None
            if family_history == "Not specified"
            else family_history == "Yes"
        ),
        "smoking_history": (
            None
            if smoking_history == "Not specified"
            else smoking_history == "Yes"
        ),
        "symptoms_duration_days": (
            None if symptoms_duration_days == 0
            else int(symptoms_duration_days)
        ),
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={
                "ct_file": (
                    ct_file.name,
                    ct_file.getvalue(),
                    "application/octet-stream",
                )
            },
            data={
                "clinical_json": json.dumps(clinical_data),
            },
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            st.success("Processing completed.")
            st.json(result)
        elif response.status_code == 503:
            st.warning(
                "The API accepted the request, but the trained prediction "
                "pipeline is not configured yet. No risk prediction was generated."
            )
            try:
                st.caption(response.json().get("detail", "Model unavailable."))
            except ValueError:
                pass
        else:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text

            st.error(f"Request failed ({response.status_code}): {detail}")

    except requests.RequestException as exc:
        st.error(f"Could not contact the prediction API: {exc}")

st.divider()
st.caption(
    "Academic research prototype • Predictions are not medical diagnoses."
)