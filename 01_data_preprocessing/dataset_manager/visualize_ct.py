from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# Project root
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


# Load CT and label
image = nib.load(IMAGE_PATH).get_fdata()
label = nib.load(LABEL_PATH).get_fdata()


# Select the middle slice
slice_index = image.shape[2] // 2

ct_slice = image[:, :, slice_index]
label_slice = label[:, :, slice_index]


# Display CT
plt.figure(figsize=(8, 8))

plt.imshow(
    np.rot90(ct_slice),
    cmap="gray"
)

plt.title(
    f"Pancreas CT - Slice {slice_index}"
)

plt.axis("off")

plt.show()


# Display segmentation
plt.figure(figsize=(8, 8))

plt.imshow(
    np.rot90(ct_slice),
    cmap="gray"
)

plt.imshow(
    np.rot90(label_slice),
    alpha=0.5,
    interpolation="none"
)

plt.title(
    f"CT + Pancreas/Cancer Segmentation - Slice {slice_index}"
)

plt.axis("off")

plt.show()