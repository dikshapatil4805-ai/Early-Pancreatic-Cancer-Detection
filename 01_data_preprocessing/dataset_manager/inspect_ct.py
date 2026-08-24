from pathlib import Path

import nibabel as nib
import numpy as np


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ct"
    / "Task07_Pancreas"
)

IMAGE_PATH = DATASET_DIR / "imagesTr" / "pancreas_001.nii.gz"
LABEL_PATH = DATASET_DIR / "labelsTr" / "pancreas_001.nii.gz"


# --------------------------------------------------
# Load CT image and label
# --------------------------------------------------

print("=" * 60)
print("PANCREAS CT SCAN INSPECTION")
print("=" * 60)

print("\nLoading CT image...")
image = nib.load(IMAGE_PATH)

print("Loading segmentation label...")
label = nib.load(LABEL_PATH)


# --------------------------------------------------
# Convert to NumPy arrays
# --------------------------------------------------

image_data = image.get_fdata()
label_data = label.get_fdata()


# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("\nCT INFORMATION")
print("-" * 40)

print("CT shape:", image_data.shape)
print("CT data type:", image_data.dtype)
print("CT minimum:", np.min(image_data))
print("CT maximum:", np.max(image_data))


print("\nLABEL INFORMATION")
print("-" * 40)

print("Label shape:", label_data.shape)
print("Label data type:", label_data.dtype)

unique_labels = np.unique(label_data)

print("Unique label values:", unique_labels)


# --------------------------------------------------
# Count each label
# --------------------------------------------------

print("\nLABEL VOXEL COUNTS")
print("-" * 40)

for label_value in unique_labels:

    count = np.sum(label_data == label_value)

    print(
        f"Label {int(label_value)}: {count:,} voxels"
    )


# --------------------------------------------------
# Verify dimensions
# --------------------------------------------------

print("\nDIMENSION CHECK")
print("-" * 40)

if image_data.shape == label_data.shape:
    print("✓ CT and label dimensions match")
else:
    print("✗ CT and label dimensions DO NOT match")


# --------------------------------------------------
# Check expected classes
# --------------------------------------------------

print("\nCLASS CHECK")
print("-" * 40)

expected_labels = {0, 1, 2}

found_labels = set(unique_labels.astype(int))

print("Expected labels:", expected_labels)
print("Found labels:", found_labels)

missing_labels = expected_labels - found_labels

if not missing_labels:
    print("✓ All expected labels are present")
else:
    print("Missing labels:", missing_labels)


print("\nInspection completed.")