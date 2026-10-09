import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from backend.schemas import ClinicalData
from backend.services.prediction_service import predict, validate_ct_file

app = FastAPI(
    title="Explainable Multimodal Pancreatic Cancer Research API",
    description=(
        "Research prototype for CT and clinical model integration. "
        "It is not a medical diagnostic system."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "project": "Early Pancreatic Cancer Detection",
        "status": "API is running",
        "prediction_ready": False,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "api_running": True,
        "models_configured": False,
    }


@app.post("/predict")
async def predict_endpoint(
    ct_file: UploadFile = File(...),
    clinical_json: str = Form(...),
):
    """
    Accept a CT file and clinical JSON.
    Prediction remains unavailable until the real models are integrated.
    """

    try:
        clinical = ClinicalData.model_validate_json(clinical_json)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_input=False),
        ) from exc

    filename = ct_file.filename or ""
    contents = await ct_file.read()

    validate_ct_file(filename, len(contents))

    clinical_data = clinical.model_dump()

    return predict(
        ct_bytes=contents,
        ct_filename=filename,
        clinical_data=clinical_data,
    )