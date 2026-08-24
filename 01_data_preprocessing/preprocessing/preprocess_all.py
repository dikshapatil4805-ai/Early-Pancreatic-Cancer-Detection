import os
import json
import nibabel as nib
import numpy as np


# ============================================================
# PATHS
# ============================================================

DATASET_JSON = "data/raw/ct/Task07_Pancreas/dataset.json"

OUTPUT_IMAGES = "data/processed/ct/images"
OUTPUT_LABELS = "data/processed/ct/labels"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)


# ============================================================
# LOAD DATASET JSON
# ============================================================

print("=" * 60)
print("PANCREAS CT BATCH PREPROCESSING")
print("=" * 60)

print("\nLoading dataset.json...")

with open(DATASET_JSON, "r") as f:
    dataset = json.load(f)

training_cases = dataset["training"]

print("Training cases found:", len(training_cases))


# ============================================================
# PROCESS EACH CASE
# ============================================================

successful = 0
failed = 0

for index, case in enumerate(training_cases, start=1):

    try:

        image_path = os.path.join(
            "data/raw/ct/Task07_Pancreas",
            case["image"].replace("./", "")
        )

        label_path = os.path.join(
            "data/raw/ct/Task07_Pancreas",
            case["label"].replace("./", "")
        )

        filename = os.path.basename(case["image"])

        print(f"\n[{index}/{len(training_cases)}] Processing {filename}")

        # ----------------------------------------------------
        # LOAD CT
        # ----------------------------------------------------

        ct_nii = nib.load(image_path)
        ct = ct_nii.get_fdata()

        # ----------------------------------------------------
        # NORMALIZE CT TO 0-1
        # Same min-max normalization used for the test case
        # ----------------------------------------------------

        ct_min = ct.min()
        ct_max = ct.max()

        if ct_max > ct_min:
            ct_processed = (ct - ct_min) / (ct_max - ct_min)
        else:
            ct_processed = np.zeros_like(ct)

        ct_processed = ct_processed.astype(np.float32)

        # ----------------------------------------------------
        # LOAD LABEL
        # ----------------------------------------------------

        label_nii = nib.load(label_path)
        label = label_nii.get_fdata()

        label = label.astype(np.uint8)

        # ----------------------------------------------------
        # CHECK DIMENSIONS
        # ----------------------------------------------------

        if ct_processed.shape != label.shape:
            raise ValueError(
                f"Shape mismatch: CT {ct_processed.shape}, "
                f"Label {label.shape}"
            )

        # ----------------------------------------------------
        # CHECK LABEL VALUES
        # ----------------------------------------------------

        unique_labels = np.unique(label)

        if not np.all(np.isin(unique_labels, [0, 1, 2])):
            raise ValueError(
                f"Invalid label values: {unique_labels}"
            )

        # ----------------------------------------------------
        # SAVE CT
        # ----------------------------------------------------

        output_ct = os.path.join(
            OUTPUT_IMAGES,
            filename
        )

        output_label = os.path.join(
            OUTPUT_LABELS,
            filename
        )

        ct_output_nii = nib.Nifti1Image(
            ct_processed,
            ct_nii.affine,
            ct_nii.header
        )

        label_output_nii = nib.Nifti1Image(
            label,
            label_nii.affine,
            label_nii.header
        )

        nib.save(ct_output_nii, output_ct)
        nib.save(label_output_nii, output_label)

        successful += 1

        print("  ✓ CT saved")
        print("  ✓ Label saved")

    except Exception as e:

        failed += 1

        print("  ✗ FAILED")
        print("  Error:", e)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 60)
print("BATCH PREPROCESSING COMPLETED")
print("=" * 60)

print("Total cases :", len(training_cases))
print("Successful  :", successful)
print("Failed      :", failed)

print("\nProcessed CT directory:")
print(os.path.abspath(OUTPUT_IMAGES))

print("\nProcessed label directory:")
print(os.path.abspath(OUTPUT_LABELS))

print("\n" + "=" * 60)