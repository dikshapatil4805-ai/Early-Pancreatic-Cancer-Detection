import os
import sys

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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
# IMPORT PROJECT FILES
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
    "confusion_matrix.png"
)


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    print("\nLoading model...")

    model = ResNet3DClassifier(
        num_classes=NUM_CLASSES
    )

    if not os.path.exists(CHECKPOINT_PATH):

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    # -----------------------------------------------------
    # Support both checkpoint formats
    # -----------------------------------------------------

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

            # Direct state dictionary
            model.load_state_dict(checkpoint)

    else:

        model.load_state_dict(checkpoint)

    model.eval()

    print("Model loaded successfully.")

    return model


# =========================================================
# GENERATE CONFUSION MATRIX
# =========================================================

def generate_confusion_matrix():

    print("=" * 55)
    print("CT MODEL - CONFUSION MATRIX")
    print("=" * 55)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # -----------------------------------------------------
    # Create DataLoaders
    # -----------------------------------------------------

    _, _, test_loader = create_dataloaders(
        batch_size=2,
        num_workers=0
    )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = load_model()
    model = model.to(device)

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    all_targets = []
    all_predictions = []

    with torch.no_grad():

        for images, targets, _ in test_loader:

            images = images.to(device)
            targets = targets.to(device)

            # Model returns:
            # logits, features

            logits, features = model(images)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_targets.extend(
                targets.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(
        all_targets,
        all_predictions,
        labels=[0, 1]
    )

    print("\nConfusion Matrix:")
    print(cm)

    # -----------------------------------------------------
    # Extract values
    # -----------------------------------------------------

    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    print("\nConfusion Matrix Details")
    print("-" * 35)

    print("TN =", tn)
    print("FP =", fp)
    print("FN =", fn)
    print("TP =", tp)

    # =====================================================
    # PLOT
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(cm)

    # Colorbar
    fig.colorbar(
        image,
        ax=ax
    )

    # Axis labels
    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    ax.set_title(
        "CT Model - Confusion Matrix"
    )

    # Tick labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["No Tumor", "Tumor"]
    )

    ax.set_yticklabels(
        ["No Tumor", "Tumor"]
    )

    # -----------------------------------------------------
    # Write values inside matrix
    # -----------------------------------------------------

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14
            )

    plt.tight_layout()

    # Save figure
    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nConfusion matrix saved at:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 55)
    print("CONFUSION MATRIX COMPLETED")
    print("=" * 55)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    generate_confusion_matrix()