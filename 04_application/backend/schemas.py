from typing import Optional

from pydantic import BaseModel, Field


class ClinicalData(BaseModel):
    """
    Clinical fields must match the features used by the trained model.
    These are illustrative generic fields, not a validated cancer-risk score.
    """

    age: int = Field(..., ge=18, le=120)
    sex: Optional[str] = None
    family_history: Optional[bool] = None
    smoking_history: Optional[bool] = None
    symptoms_duration_days: Optional[int] = Field(
        default=None, ge=0, le=36500
    )


class PredictionResponse(BaseModel):
    status: str
    message: str
    prediction: Optional[str] = None
    risk_probability: Optional[float] = None
    explanation: Optional[dict] = None