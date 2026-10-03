"""
Multimodal inference interface for PDAC risk prediction.

This module connects:
    clinical data -> clinical preprocessing -> clinical feature vector
    CT feature vector -> multimodal fusion -> risk prediction

Important:
    The current repository does not contain verified paired PANORAMA
    clinical records and Task07 Pancreas CT cases. Therefore this module
    provides the inference interface and architecture test only.

    The self-test uses a dummy 512-dimensional CT feature vector.
    It must not be interpreted as a real multimodal prediction.
"""

from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import torch

from clinical_model.model import ClinicalMLP
from fusion.concatenation import ConcatenationFusion
from fusion.cross_attention import CrossAttentionFusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAVED_MODELS_DIR = (
    PROJECT_ROOT
    / "03_multimodal_fusion"
    / "saved_models"
)

CLINICAL_PREPROCESSOR_PATH = (
    SAVED_MODELS_DIR
    / "clinical_preprocessor.joblib"
)

CLINICAL_MODEL_PATH = (
    SAVED_MODELS_DIR
    / "clinical_mlp.pth"
)


CLINICAL_INPUT_DIM = 13
CLINICAL_FEATURE_DIM = 16
CT_FEATURE_DIM = 512


def load_clinical_components():
    """
    Load the saved clinical preprocessor and trained clinical MLP.
    """

    preprocessor = joblib.load(
        CLINICAL_PREPROCESSOR_PATH
    )

    clinical_model = ClinicalMLP(
        input_dim=CLINICAL_INPUT_DIM
    )

    state_dict = torch.load(
        CLINICAL_MODEL_PATH,
        map_location="cpu",
    )

    clinical_model.load_state_dict(
        state_dict
    )

    clinical_model.eval()

    return preprocessor, clinical_model


def extract_clinical_features(
    clinical_data: pd.DataFrame,
    preprocessor,
    clinical_model,
) -> torch.Tensor:
    """
    Convert raw clinical data into the 16-dimensional
    representation learned by the clinical MLP.

    Parameters
    ----------
    clinical_data : pandas.DataFrame
        Raw clinical input containing the required clinical columns.

    preprocessor :
        Fitted clinical preprocessing pipeline.

    clinical_model : ClinicalMLP
        Trained clinical model.

    Returns
    -------
    torch.Tensor
        Clinical representation with shape:
        [batch_size, 16]
    """

    transformed = preprocessor.transform(
        clinical_data
    )

    transformed_tensor = torch.tensor(
        np.asarray(
            transformed,
            dtype=np.float32,
        ),
        dtype=torch.float32,
    )

    with torch.no_grad():

        hidden_features = clinical_model.network(
            transformed_tensor
        )

        # The complete network ends with the final
        # 1-unit classifier. We need the 16-D layer
        # immediately before that classifier.
        clinical_features = clinical_model.network[
            :-1
        ](
            transformed_tensor
        )

    if clinical_features.ndim != 2:
        raise ValueError(
            "Clinical features must have shape "
            "[batch_size, 16]."
        )

    if clinical_features.size(1) != CLINICAL_FEATURE_DIM:
        raise ValueError(
            f"Expected clinical feature dimension "
            f"{CLINICAL_FEATURE_DIM}, got "
            f"{clinical_features.size(1)}."
        )

    return clinical_features


def predict_multimodal(
    ct_features,
    clinical_data: pd.DataFrame,
    fusion_type: str = "concatenation",
) -> Dict[str, Any]:
    """
    Generate a multimodal PDAC risk prediction.

    Parameters
    ----------
    ct_features : array-like or torch.Tensor
        Precomputed CT feature vector(s).

        Expected shape:
            [512] for one sample
            [batch_size, 512] for multiple samples

    clinical_data : pandas.DataFrame
        Raw clinical data.

    fusion_type : str
        Either:
            "concatenation"
            "cross_attention"

    Returns
    -------
    dict
        Contains risk probabilities, predictions,
        CT features, and clinical features.

    Notes
    -----
    The CT feature vector must come from the CT model.
    This function does not perform CT image preprocessing
    or CT feature extraction.
    """

    preprocessor, clinical_model = (
        load_clinical_components()
    )

    clinical_features = extract_clinical_features(
        clinical_data,
        preprocessor,
        clinical_model,
    )

    if isinstance(ct_features, np.ndarray):

        ct_features = torch.tensor(
            ct_features,
            dtype=torch.float32,
        )

    elif not isinstance(
        ct_features,
        torch.Tensor,
    ):

        ct_features = torch.tensor(
            ct_features,
            dtype=torch.float32,
        )

    if ct_features.ndim == 1:
        ct_features = ct_features.unsqueeze(0)

    if ct_features.ndim != 2:
        raise ValueError(
            "CT features must have shape "
            "[batch_size, 512]."
        )

    if ct_features.size(1) != CT_FEATURE_DIM:
        raise ValueError(
            f"Expected CT feature dimension "
            f"{CT_FEATURE_DIM}, got "
            f"{ct_features.size(1)}."
        )

    if ct_features.size(0) != clinical_features.size(0):
        raise ValueError(
            "CT and clinical inputs must contain "
            "the same number of samples."
        )

    if fusion_type == "concatenation":

        fusion_model = ConcatenationFusion(
            ct_feature_dim=CT_FEATURE_DIM,
            clinical_feature_dim=CLINICAL_FEATURE_DIM,
        )

    elif fusion_type == "cross_attention":

        fusion_model = CrossAttentionFusion(
            ct_feature_dim=CT_FEATURE_DIM,
            clinical_feature_dim=CLINICAL_FEATURE_DIM,
        )

    else:

        raise ValueError(
            "fusion_type must be either "
            "'concatenation' or 'cross_attention'."
        )

    fusion_model.eval()

    with torch.no_grad():

        logits, fused_features = fusion_model(
            ct_features,
            clinical_features,
        )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).long()

    return {
        "risk_probability": (
            probabilities.cpu().numpy()
        ),
        "prediction": (
            predictions.cpu().numpy()
        ),
        "ct_features": (
            ct_features.cpu().numpy()
        ),
        "clinical_features": (
            clinical_features.cpu().numpy()
        ),
        "fused_features": (
            fused_features.cpu().numpy()
        ),
        "fusion_type": fusion_type,
    }


def main():
    """
    Architecture/interface self-test.

    Uses a dummy CT feature vector and one real
    clinical record from the processed clinical data.
    This is NOT a validated multimodal prediction.
    """

    print("=" * 60)
    print("MULTIMODAL INFERENCE INTERFACE TEST")
    print("=" * 60)

    clinical_csv = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "clinical"
        / "test.csv"
    )

    clinical_data = pd.read_csv(
        clinical_csv
    ).iloc[[0]].copy()

    print()
    print(
        "Clinical input shape:",
        clinical_data.shape,
    )

    # Dummy CT representation for interface testing only.
    dummy_ct_features = np.zeros(
        (1, CT_FEATURE_DIM),
        dtype=np.float32,
    )

    print(
        "Dummy CT feature shape:",
        dummy_ct_features.shape,
    )

    for fusion_type in [
        "concatenation",
        "cross_attention",
    ]:

        print()
        print(
            f"Testing fusion: {fusion_type}"
        )

        result = predict_multimodal(
            ct_features=dummy_ct_features,
            clinical_data=clinical_data,
            fusion_type=fusion_type,
        )

        print(
            "Clinical feature shape:",
            result["clinical_features"].shape,
        )

        print(
            "Fused feature shape:",
            result["fused_features"].shape,
        )

        print(
            "Risk probability:",
            result["risk_probability"],
        )

        print(
            "Prediction:",
            result["prediction"],
        )

    print()
    print(
        "Interface test completed."
    )
    print(
        "WARNING: CT features were dummy "
        "zeros; results are not real multimodal "
        "predictions."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()