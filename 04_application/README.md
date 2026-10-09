# Early Pancreatic Cancer Detection

## Project Title

Early Pancreatic Cancer Detection Using Explainable Multimodal Deep Learning

## Overview

This academic research prototype is designed to explore the integration of CT medical imaging and structured clinical information for pancreatic cancer research.

## Application Components

* **FastAPI backend:** Handles API requests and input validation.
* **Streamlit frontend:** Provides the research interface.
* **CT processing:** Intended to use the team's CT preprocessing and inference modules.
* **Clinical model:** Must be integrated using the verified clinical dataset and trained model.
* **Multimodal fusion:** Must combine compatible CT and clinical features.
* **Explainability:** Grad-CAM and SHAP can be integrated once the relevant models support them.

## Current Implementation Status

The API and frontend provide a starter application. Actual risk prediction is not enabled until the trained CT model, clinical model, and fusion pipeline have been connected and validated.

## Setup

From the repository root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the backend from the application directory:

```bash
cd 04_application
python -m uvicorn backend.main:app --reload
```

Open the API documentation at `http://127.0.0.1:8000/docs`.

In a second terminal, from the repository root, start the frontend:

```bash
python -m streamlit run 04_application/frontend/app.py
```

The frontend normally opens at `http://localhost:8501`.

## Tests

From the repository root:

```bash
cd 04_application
python -m pytest tests/test_api.py -v
```

## Important Notice

This application is for academic research and software testing only. It is not a validated medical diagnostic system and must not be used for patient-care decisions.

Do not commit patient-identifiable information, raw medical datasets, API keys, or confidential credentials to GitHub.

