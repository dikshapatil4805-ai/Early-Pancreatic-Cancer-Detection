import os
import nibabel as nib
import numpy as np


# ============================================================
# PANCREAS CT DATASET LOADER
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "ct",
    "images"
)

LABEL_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "ct",
    "labels"
)


def load_ct_case(case_name):
    """
    Load one processed CT volume and its segmentation label.
    """

    image_path = os.path.join(IMAGE_DIR, case_name)
    label_path = os.path.join(LABEL_DIR, case_name)

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"CT image not found: {image_path}"
        )

    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"Label not found: {label_path}"
        )

    print("Loading CT:")
    print(image_path)

    print("\nLoading label:")
    print(label_path)

    ct_nii = nib.load(image_path)
    label_nii = nib.load(label_path)

    ct = ct_nii.get_fdata().astype(np.float32)
    label = label_nii.get_fdata().astype(np.uint8)

    return ct, label


def inspect_dataset():
    """
    Inspect the processed CT dataset.
    """

    print("=" * 60)
    print("PANCREAS CT DATASET LOADER")
    print("=" * 60)

    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    if not os.path.exists(LABEL_DIR):
        raise FileNotFoundError(
            f"Label directory not found: {LABEL_DIR}"
        )

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.endswith(".nii.gz")
    ]

    label_files = [
        f for f in os.listdir(LABEL_DIR)
        if f.endswith(".nii.gz")
    ]

    print("\nDATASET INFORMATION")
    print("-" * 40)

    print(f"CT volumes found: {len(image_files)}")
    print(f"Label volumes found: {len(label_files)}")

    if len(image_files) == 0:
        print("\nNo CT volumes found.")
        return

    print("\nCT FILES")
    print("-" * 40)

    for filename in image_files:
        print(filename)

    print("\nLABEL FILES")
    print("-" * 40)

    for filename in label_files:
        print(filename)

    # Test first CT case
    case_name = image_files[0]

    print("\nTESTING FIRST CASE")
    print("-" * 40)

    ct, label = load_ct_case(case_name)

    print("\nCT INFORMATION")
    print("-" * 40)
    print(f"Shape: {ct.shape}")
    print(f"Data type: {ct.dtype}")
    print(f"Minimum: {ct.min()}")
    print(f"Maximum: {ct.max()}")

    print("\nLABEL INFORMATION")
    print("-" * 40)
    print(f"Shape: {label.shape}")
    print(f"Data type: {label.dtype}")
    print(f"Unique values: {np.unique(label)}")

    print("\nDIMENSION CHECK")
    print("-" * 40)

    if ct.shape == label.shape:
        print("✓ CT and label dimensions match")
    else:
        print("✗ CT and label dimensions DO NOT match")

    print("\nNORMALIZATION CHECK")
    print("-" * 40)

    if ct.min() >= 0.0 and ct.max() <= 1.0:
        print("✓ CT values are normalized between 0 and 1")
    else:
        print("✗ CT values are outside expected range")

    print("\nLABEL CHECK")
    print("-" * 40)

    expected_labels = {0, 1, 2}
    found_labels = set(np.unique(label).tolist())

    print(f"Expected labels: {expected_labels}")
    print(f"Found labels: {found_labels}")

    if found_labels.issubset(expected_labels):
        print("✓ Label values are valid")
    else:
        print("✗ Unexpected label values found")

    print("\n" + "=" * 60)
    print("DATASET LOADER TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset()