import sys
import os
import torch
from torch.utils.data import DataLoader


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CT_MODEL_DIR = os.path.dirname(CURRENT_DIR)

DATASETS_DIR = os.path.join(
    CT_MODEL_DIR,
    "datasets"
)

MODELS_DIR = os.path.join(
    CT_MODEL_DIR,
    "models"
)

# Add folders containing our Python files
sys.path.append(DATASETS_DIR)
sys.path.append(MODELS_DIR)


# ============================================================
# IMPORT DATASET AND MODEL
# ============================================================

from ct_segmentation_dataset import PancreasCTDataset
from unet3d import UNet3D


# ============================================================
# TEST DATALOADER AND MODEL
# ============================================================

print("=" * 60)
print("CT DATALOADER AND MODEL TEST")
print("=" * 60)


# Load training dataset
dataset = PancreasCTDataset(
    split="train"
)


# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)


print()
print(f"Training cases: {len(dataset)}")


# ============================================================
# LOAD ONE REAL BATCH
# ============================================================

images, labels = next(iter(dataloader))


print()
print("BATCH INFORMATION")
print("-" * 40)

print(f"CT batch shape: {images.shape}")
print(f"CT data type: {images.dtype}")

print()

print(f"Label batch shape: {labels.shape}")
print(f"Label data type: {labels.dtype}")

print(f"Unique labels: {torch.unique(labels)}")


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("CREATING MODEL")
print("-" * 40)

model = UNet3D(
    in_channels=1,
    num_classes=3
)

model.eval()


# ============================================================
# FORWARD PASS
# ============================================================

print()
print("RUNNING FORWARD PASS")
print("-" * 40)


with torch.no_grad():
    outputs = model(images)


print()

print(f"Model output shape: {outputs.shape}")

print(
    "Expected shape: "
    "[1, 3, 64, 64, 64]"
)


# ============================================================
# CHECK PREDICTIONS
# ============================================================

predictions = torch.argmax(
    outputs,
    dim=1
)


print()

print("PREDICTION INFORMATION")
print("-" * 40)

print(f"Prediction shape: {predictions.shape}")

print(
    f"Unique predicted classes: "
    f"{torch.unique(predictions)}"
)


print()
print("=" * 60)
print("DATALOADER AND MODEL TEST COMPLETED")
print("=" * 60)