import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

# =========================================================
# PATH SETUP
# =========================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_DIR)

# =========================================================
# IMPORTS
# =========================================================

from data_loader import create_dataloaders
from models.resnet_model import ResNet3DClassifier

# =========================================================
# CONFIGURATION
# =========================================================

NUM_CLASSES = 2

CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "best_resnet3d.pth"
)

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "roc_curve.png"
)

# =========================================================
# LOAD MODEL
# =========================================================

def load_model(device):

    model = ResNet3DClassifier(
        num_classes=NUM_CLASSES
    )

    if not os.path.exists(CHECKPOINT_PATH):

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["state_dict"]
            )

        else:

            model.load_state_dict(checkpoint)

    else:

        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


# =========================================================
# COLLECT TEST PREDICTIONS
# =========================================================

def get_test_predictions(model, test_loader, device):

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels, case_names in test_loader:

            images = images.to(device)

            logits, features = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            tumor_probability = probabilities[:, 1]

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                tumor_probability.cpu().numpy().tolist()
            )

    return (
        np.array(all_labels),
        np.array(all_probabilities)
    )


# =========================================================
# CREATE ROC CURVE
# =========================================================

def generate_roc_curve():

    print("=" * 60)
    print("MEMBER 2 - CT MODEL ROC CURVE")
    print("=" * 60)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    # -----------------------------------------------------
    # Load Data
    # -----------------------------------------------------

    (
        train_loader,
        validation_loader,
        test_loader
    ) = create_dataloaders(
        batch_size=2,
        num_workers=0
    )

    # -----------------------------------------------------
    # Load Model
    # -----------------------------------------------------

    print("\nLoading model...")

    model = load_model(device)

    print("Model loaded successfully.")

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    print("\nGenerating test predictions...")

    y_true, y_score = get_test_predictions(
        model,
        test_loader,
        device
    )

    print(
        "Number of test samples:",
        len(y_true)
    )

    # -----------------------------------------------------
    # Class Distribution
    # -----------------------------------------------------

    unique_classes = np.unique(y_true)

    print("\nTest Set Classes:")
    print(unique_classes)

    print(
        "No Tumor samples:",
        np.sum(y_true == 0)
    )

    print(
        "Tumor samples:",
        np.sum(y_true == 1)
    )

    # -----------------------------------------------------
    # Check if both classes exist
    # -----------------------------------------------------

    if len(unique_classes) < 2:

        print("\n" + "=" * 60)
        print("ROC CURVE CANNOT BE CALCULATED")
        print("=" * 60)

        print(
            "\nReason:"
        )

        print(
            "The test set contains only one class."
        )

        print(
            "ROC/AUROC requires both positive and "
            "negative classes."
        )

        print(
            "\nCurrent test set:"
        )

        print(
            f"No Tumor = {np.sum(y_true == 0)}"
        )

        print(
            f"Tumor    = {np.sum(y_true == 1)}"
        )

        print(
            "\nNo fake ROC curve or AUROC value will "
            "be generated."
        )

        print("\n" + "=" * 60)
        print("ROC CURVE TEST COMPLETED WITH LIMITATION")
        print("=" * 60)

        return

    # -----------------------------------------------------
    # Calculate ROC
    # -----------------------------------------------------

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    # -----------------------------------------------------
    # Plot ROC
    # -----------------------------------------------------

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC curve (AUROC = {roc_auc:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1
    )

    plt.xlim(
        [0.0, 1.0]
    )

    plt.ylim(
        [0.0, 1.05]
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "CT Model ROC Curve"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------------------------------
    # Print Results
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("ROC CURVE RESULTS")
    print("=" * 60)

    print(
        f"\nAUROC : {roc_auc:.4f}"
    )

    print(
        "\nROC curve saved at:"
    )

    print(
        OUTPUT_PATH
    )

    print("\n" + "=" * 60)
    print("ROC CURVE COMPLETED SUCCESSFULLY")
    print("=" * 60)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    generate_roc_curve()