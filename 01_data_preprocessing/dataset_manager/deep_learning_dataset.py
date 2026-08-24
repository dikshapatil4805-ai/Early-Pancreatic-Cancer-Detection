import os
import nibabel as nib
import numpy as np


# ============================================================
# PANCREAS CT DEEP LEARNING DATASET LOADER
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ct",
    "images"
)

LABEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ct",
    "labels"
)

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ct",
    "splits"
)


# ------------------------------------------------------------
# LOAD SPLIT FILE
# ------------------------------------------------------------

def load_split(split_name):

    split_path = os.path.join(
        SPLIT_DIR,
        split_name + ".txt"
    )

    with open(split_path, "r") as f:

        cases = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return cases


# ------------------------------------------------------------
# LOAD ONE CASE
# ------------------------------------------------------------

def load_case(case_name):

    image_path = os.path.join(
        IMAGE_DIR,
        case_name
    )

    label_path = os.path.join(
        LABEL_DIR,
        case_name
    )


    # Load CT image
    ct_image = nib.load(image_path)

    ct_data = ct_image.get_fdata().astype(
        np.float32
    )


    # Load segmentation label
    label_image = nib.load(label_path)

    label_data = label_image.get_fdata().astype(
        np.uint8
    )


    return ct_data, label_data


# ------------------------------------------------------------
# TEST DATASET LOADER
# ------------------------------------------------------------

print("=" * 60)
print("PANCREAS CT DEEP LEARNING DATASET LOADER")
print("=" * 60)


# Load training split
train_cases = load_split("train")

print()
print("TRAINING DATASET")
print("-" * 40)

print(f"Training cases: {len(train_cases)}")


# Select first case
test_case = train_cases[0]

print()
print(f"Loading case: {test_case}")


# Load CT and label
ct, label = load_case(test_case)


# ------------------------------------------------------------
# DISPLAY INFORMATION
# ------------------------------------------------------------

print()
print("CT INFORMATION")
print("-" * 40)

print(f"Shape: {ct.shape}")
print(f"Data type: {ct.dtype}")
print(f"Minimum: {ct.min()}")
print(f"Maximum: {ct.max()}")


print()
print("LABEL INFORMATION")
print("-" * 40)

print(f"Shape: {label.shape}")
print(f"Data type: {label.dtype}")
print(f"Unique labels: {np.unique(label)}")


# ------------------------------------------------------------
# CHECK DIMENSIONS
# ------------------------------------------------------------

print()
print("DIMENSION CHECK")
print("-" * 40)

if ct.shape == label.shape:

    print("✓ CT and label dimensions match")

else:

    print("✗ ERROR: CT and label dimensions do not match")


print()
print("=" * 60)
print("DEEP LEARNING DATASET LOADER COMPLETED")
print("=" * 60)