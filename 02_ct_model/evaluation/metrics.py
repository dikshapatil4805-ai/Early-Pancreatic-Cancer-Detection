import sys
from pathlib import Path

import torch
import numpy as np

# Add 02_ct_model to Python path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data_loader import create_dataloaders
from models.resnet_model import ResNet3DClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 2

MODEL_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "best_resnet3d.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("MEMBER 2 - CT RESNET EVALUATION")
print("=" * 60)

print("Device:", device)


# ============================================================
# LOAD TEST DATA
# ============================================================

_, _, test_loader = create_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=0
)


# ============================================================
# LOAD MODEL
# ============================================================

model = ResNet3DClassifier(
    num_classes=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Support both complete checkpoint and state_dict
if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

all_targets = []
all_predictions = []
all_probabilities = []


with torch.no_grad():

    for images, targets, _ in test_loader:

        images = images.to(device)
        targets = targets.to(device)

        logits, _ = model(images)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_targets.extend(
            targets.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Probability of positive class
        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

y_true = np.array(all_targets)
y_pred = np.array(all_predictions)
y_prob = np.array(all_probabilities)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

TN, FP, FN, TP = cm.ravel()

# Sensitivity = TP / (TP + FN)
sensitivity = (
    TP / (TP + FN)
    if (TP + FN) > 0
    else 0
)

# Specificity = TN / (TN + FP)
specificity = (
    TN / (TN + FP)
    if (TN + FP) > 0
    else 0
)


# ============================================================
# AUROC
# ============================================================

try:

    auroc = roc_auc_score(
        y_true,
        y_prob
    )

except ValueError:

    auroc = float("nan")


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 60)
print("TEST SET RESULTS")
print("=" * 60)

print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"Sensitivity  : {sensitivity:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"F1-Score     : {f1:.4f}")
print(f"AUROC        : {auroc:.4f}")

print()
print("Confusion Matrix:")
print(cm)

print()
print(f"TN = {TN}")
print(f"FP = {FP}")
print(f"FN = {FN}")
print(f"TP = {TP}")

print("=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)