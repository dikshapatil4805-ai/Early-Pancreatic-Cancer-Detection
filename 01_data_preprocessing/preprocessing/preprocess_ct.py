import os
import numpy as np
import nibabel as nib


# ============================================================
# PANCREAS CT PREPROCESSING
# ============================================================

# Project folders
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RAW_DATA = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "ct",
    "Task07_Pancreas"
)

OUTPUT_DATA = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "ct"
)


# Create output folders
os.makedirs(os.path.join(OUTPUT_DATA, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DATA, "labels"), exist_ok=True)


# ------------------------------------------------------------
# Select one CT scan for testing
# ------------------------------------------------------------

image_name = "pancreas_001.nii.gz"

image_path = os.path.join(
    RAW_DATA,
    "imagesTr",
    image_name
)

label_path = os.path.join(
    RAW_DATA,
    "labelsTr",
    image_name
)


print("=" * 60)
print("PANCREAS CT PREPROCESSING")
print("=" * 60)

print("\nLoading CT image...")
ct_nii = nib.load(image_path)
ct = ct_nii.get_fdata()

print("Loading segmentation label...")
label_nii = nib.load(label_path)
label = label_nii.get_fdata()


# ------------------------------------------------------------
# Display original information
# ------------------------------------------------------------

print("\nORIGINAL CT")
print("-" * 40)
print("Shape:", ct.shape)
print("Minimum:", ct.min())
print("Maximum:", ct.max())


# ------------------------------------------------------------
# CT normalization
# ------------------------------------------------------------

# Clip CT values to a reasonable range
ct = np.clip(ct, -1000, 400)

# Normalize values to 0-1
ct = (ct + 1000) / 1400

# Convert to float32 to reduce storage size
ct = ct.astype(np.float32)


# ------------------------------------------------------------
# Check labels
# ------------------------------------------------------------

label = label.astype(np.uint8)

unique_labels = np.unique(label)

print("\nPROCESSED CT")
print("-" * 40)
print("Shape:", ct.shape)
print("Minimum:", ct.min())
print("Maximum:", ct.max())
print("Data type:", ct.dtype)

print("\nLABEL")
print("-" * 40)
print("Shape:", label.shape)
print("Unique values:", unique_labels)


# ------------------------------------------------------------
# Verify dimensions
# ------------------------------------------------------------

if ct.shape != label.shape:
    raise ValueError("CT and label dimensions do not match!")

print("\n✓ CT and label dimensions match")


# ------------------------------------------------------------
# Verify label classes
# ------------------------------------------------------------

expected_labels = {0, 1, 2}
found_labels = set(unique_labels.tolist())

if not found_labels.issubset(expected_labels):
    raise ValueError(
        f"Unexpected label values found: {found_labels}"
    )

print("✓ Label values are valid")


# ------------------------------------------------------------
# Save processed files
# ------------------------------------------------------------

output_image = os.path.join(
    OUTPUT_DATA,
    "images",
    image_name
)

output_label = os.path.join(
    OUTPUT_DATA,
    "labels",
    image_name
)


# Preserve original NIfTI spatial information
processed_ct_nii = nib.Nifti1Image(
    ct,
    ct_nii.affine,
    ct_nii.header
)

processed_label_nii = nib.Nifti1Image(
    label,
    label_nii.affine,
    label_nii.header
)


nib.save(processed_ct_nii, output_image)
nib.save(processed_label_nii, output_label)


print("\nOUTPUT")
print("-" * 40)
print("Processed CT saved to:")
print(output_image)

print("\nProcessed label saved to:")
print(output_label)

print("\n" + "=" * 60)
print("PREPROCESSING TEST COMPLETED SUCCESSFULLY")
print("=" * 60)