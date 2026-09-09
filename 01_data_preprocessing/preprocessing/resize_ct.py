import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# ============================================================
# PANCREAS CT DATASET RESIZING
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

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

OUTPUT_IMAGE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ct",
    "resized_images"
)

OUTPUT_LABEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ct",
    "resized_labels"
)

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)


# ============================================================
# RESIZE FUNCTION
# ============================================================

def resize_volume(volume, target_shape, order):

    factors = [
        target_shape[i] / volume.shape[i]
        for i in range(3)
    ]

    resized = zoom(
        volume,
        factors,
        order=order
    )

    return resized


# ============================================================
# FIND ALL CASES
# ============================================================

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".nii.gz")
])

print("=" * 60)
print("PANCREAS CT DATASET RESIZING")
print("=" * 60)

print()
print(f"Total CT volumes found: {len(image_files)}")
print()


# ============================================================
# PROCESS ALL CASES
# ============================================================

TARGET_SHAPE = (64, 64, 64)

successful = 0

for index, filename in enumerate(image_files):

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    label_path = os.path.join(
        LABEL_DIR,
        filename
    )

    # Skip if matching label does not exist
    if not os.path.exists(label_path):

        print(
            f"[SKIPPED] No label found for {filename}"
        )

        continue

    print(
        f"[{index + 1}/{len(image_files)}] "
        f"Processing: {filename}"
    )

    # Load CT
    image_nii = nib.load(image_path)

    image = image_nii.get_fdata().astype(
        np.float32
    )

    # Load label
    label_nii = nib.load(label_path)

    label = label_nii.get_fdata().astype(
        np.uint8
    )

    # Resize CT
    resized_image = resize_volume(
        image,
        TARGET_SHAPE,
        order=1
    ).astype(np.float32)

    # Resize label
    resized_label = resize_volume(
        label,
        TARGET_SHAPE,
        order=0
    ).astype(np.uint8)

    # Save CT
    output_image_path = os.path.join(
        OUTPUT_IMAGE_DIR,
        filename
    )

    nib.save(
        nib.Nifti1Image(
            resized_image,
            image_nii.affine
        ),
        output_image_path
    )

    # Save label
    output_label_path = os.path.join(
        OUTPUT_LABEL_DIR,
        filename
    )

    nib.save(
        nib.Nifti1Image(
            resized_label,
            label_nii.affine
        ),
        output_label_path
    )

    successful += 1

    print(
        f"    ✓ Resized to {TARGET_SHAPE}"
    )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 60)
print("DATASET RESIZING COMPLETED")
print("=" * 60)

print()
print(f"Successfully processed: {successful} cases")

print()
print("Resized CT images:")
print(OUTPUT_IMAGE_DIR)

print()
print("Resized labels:")
print(OUTPUT_LABEL_DIR)