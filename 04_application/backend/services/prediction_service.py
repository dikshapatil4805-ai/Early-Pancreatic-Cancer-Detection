from pathlib import Path
from typing import Any

from fastapi import HTTPException

SUPPORTED_EXTENSIONS = (
    ".dcm",
    ".nii",
    ".nii.gz",
    ".zip",
)


def validate_ct_file(filename: str, file_size: int) -> None:
    """Validate the filename and upload size."""

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Please select a CT scan file.",
        )

    normalized_name = filename.lower()

    if not any(
        normalized_name.endswith(ext)
        for ext in SUPPORTED_EXTENSIONS
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file. Use DICOM, NIfTI, or a ZIP archive.",
        )

    if file_size == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    max_size = 100 * 1024 * 1024

    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 100 MB upload limit.",
        )


def predict(
    ct_bytes: bytes,
    ct_filename: str,
    clinical_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Model integration point.

    Do not return a fabricated prediction. Connect the trained CT model,
    clinical model, and fusion model here after their interfaces are verified.
    """

    raise HTTPException(
        status_code=503,
        detail=(
            "Prediction is not available yet. The trained CT model, "
            "clinical model, and multimodal fusion pipeline must be "
            "configured and validated first."
        ),
    )