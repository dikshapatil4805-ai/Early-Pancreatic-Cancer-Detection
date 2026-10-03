"""
SHAP explainability for the clinical PDAC baseline model.

This module explains how the transformed clinical features
contribute to the clinical-only PDAC risk prediction.

Important:
- Explanations are for the clinical model only.
- No CT/clinical patient pairing is required.
- The existing clinical preprocessing pipeline is reused.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(
    __file__
).resolve().parents[1]

CLINICAL_MODEL_DIR = (
    PROJECT_DIR / "clinical_model"
)

sys.path.insert(
    0,
    str(CLINICAL_MODEL_DIR)
)

from data_loader import prepare_clinical_data
from model import ClinicalMLP


MODEL_PATH = (
    PROJECT_DIR
    / "saved_models"
    / "clinical_mlp.pth"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------------------------
# Load trained clinical model
# ---------------------------------------------------------------------------

def load_clinical_model(
    input_dim: int,
) -> ClinicalMLP:
    """
    Load the trained ClinicalMLP model.
    """

    model = ClinicalMLP(
        input_dim=input_dim
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    return model


# ---------------------------------------------------------------------------
# Prediction function for SHAP
# ---------------------------------------------------------------------------

def predict_probability(
    model: ClinicalMLP,
    features: np.ndarray,
) -> np.ndarray:
    """
    Convert model logits into PDAC probabilities.
    """

    features = np.asarray(
        features,
        dtype=np.float32
    )

    tensor = torch.from_numpy(
        features
    )

    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = torch.sigmoid(
            logits
        )

    return probabilities.numpy()


# ---------------------------------------------------------------------------
# Main SHAP analysis
# ---------------------------------------------------------------------------

def run_shap_analysis():

    print("=" * 60)
    print("CLINICAL MODEL SHAP EXPLAINABILITY")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Load the repository's existing preprocessing pipeline.
    # -----------------------------------------------------------------------

    (
        preprocessor,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = prepare_clinical_data()

    feature_names = list(
        X_train.columns
    )

    print()
    print(
        "Training feature shape:",
        X_train.shape
    )

    print(
        "Validation feature shape:",
        X_validation.shape
    )

    print(
        "Test feature shape:",
        X_test.shape
    )

    print(
        "Number of features:",
        len(feature_names)
    )

    # -----------------------------------------------------------------------
    # Load trained model.
    # -----------------------------------------------------------------------

    model = load_clinical_model(
        input_dim=X_train.shape[1]
    )

    print()
    print("Clinical model loaded.")

    # -----------------------------------------------------------------------
    # Select SHAP background data.
    # -----------------------------------------------------------------------

    background_size = min(
        100,
        len(X_train)
    )

    background = X_train.iloc[
        :background_size
    ].to_numpy(
        dtype=np.float32
    )

    # Explain a manageable subset of the test set.
    explain_size = min(
        100,
        len(X_test)
    )

    X_explain = X_test.iloc[
        :explain_size
    ].to_numpy(
        dtype=np.float32
    )

    print()
    print(
        "SHAP background shape:",
        background.shape
    )

    print(
        "Samples explained:",
        X_explain.shape[0]
    )

    # -----------------------------------------------------------------------
    # SHAP explainer.
    # -----------------------------------------------------------------------

    def model_function(x):

        return predict_probability(
            model,
            x
        )

    explainer = shap.Explainer(
        model_function,
        background
    )

    shap_values = explainer(
        X_explain
    )

    # -----------------------------------------------------------------------
    # Feature importance.
    # -----------------------------------------------------------------------

    mean_abs_shap = np.abs(
        shap_values.values
    ).mean(
        axis=0
    )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    )

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False
    )

    importance_path = (
        RESULTS_DIR
        / "clinical_shap_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False
    )

    # -----------------------------------------------------------------------
    # Save raw SHAP values.
    # -----------------------------------------------------------------------

    shap_values_path = (
        RESULTS_DIR
        / "clinical_shap_values.npy"
    )

    np.save(
        shap_values_path,
        shap_values.values
    )

    # -----------------------------------------------------------------------
    # SHAP summary plot.
    # -----------------------------------------------------------------------

    summary_path = (
        RESULTS_DIR
        / "clinical_shap_summary.png"
    )

    shap.summary_plot(
        shap_values.values,
        X_explain,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        summary_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------------------------------------------------
    # Print feature importance.
    # -----------------------------------------------------------------------

    print()
    print(
        "Top clinical features by mean |SHAP|:"
    )

    print("-" * 60)

    for _, row in importance.head(10).iterrows():

        print(
            f"{row['feature']:<45}"
            f"{row['mean_abs_shap']:.6f}"
        )

    print()
    print("Saved SHAP artifacts:")
    print(importance_path)
    print(shap_values_path)
    print(summary_path)

    print()
    print(
        "Clinical SHAP analysis completed."
    )

    print("=" * 60)


if __name__ == "__main__":

    run_shap_analysis()