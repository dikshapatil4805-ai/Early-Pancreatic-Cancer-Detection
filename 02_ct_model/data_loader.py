import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    ScaleIntensity,
    EnsureType,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CT_IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "ct" / "resized_images"
CT_LABEL_DIR = PROJECT_ROOT / "data" / "processed" / "ct" / "resized_labels"
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "ct" / "splits"


# ============================================================
# TRANSFORMS
# ============================================================

image_transform = Compose([
    EnsureChannelFirst(channel_dim="no_channel"),
    ScaleIntensity(),
    EnsureType(),
])


# ============================================================
# CT DATASET
# ============================================================

class CTClassificationDataset(Dataset):

    def __init__(self, split="train"):

        self.split = split

        self.image_dir = CT_IMAGE_DIR
        self.label_dir = CT_LABEL_DIR

        split_file = SPLIT_DIR / f"{split}.txt"

        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}"
            )

        with open(split_file, "r") as f:
            self.case_names = [
                line.strip()
                for line in f
                if line.strip()
            ]

        print(f"{split.upper()} dataset: {len(self.case_names)} cases")

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, index):

        case_name = self.case_names[index]

        image_path = self.image_dir / case_name
        label_path = self.label_dir / case_name

        if not image_path.exists():
            raise FileNotFoundError(
                f"CT image not found: {image_path}"
            )

        if not label_path.exists():
            raise FileNotFoundError(
                f"CT label not found: {label_path}"
            )

        # ----------------------------------------------------
        # Load CT
        # ----------------------------------------------------

        image_nii = nib.load(str(image_path))
        image = image_nii.get_fdata().astype(np.float32)

        # ----------------------------------------------------
        # Load segmentation label
        # ----------------------------------------------------

        label_nii = nib.load(str(label_path))
        label = label_nii.get_fdata()

        # ----------------------------------------------------
        # Shape check
        # ----------------------------------------------------

        if image.shape != label.shape:
            raise ValueError(
                f"Shape mismatch for {case_name}: "
                f"image={image.shape}, label={label.shape}"
            )

        # ----------------------------------------------------
        # Convert segmentation label to classification target
        #
        # Label 2 = pancreatic tumor
        # Tumor present  -> 1
        # Tumor absent   -> 0
        # ----------------------------------------------------

        if np.any(label == 2):
            target = 1
        else:
            target = 0

        # ----------------------------------------------------
        # Convert image to tensor
        # ----------------------------------------------------

        image = image_transform(image)

        target = torch.tensor(
            target,
            dtype=torch.long
        )

        return image, target, case_name


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    batch_size=2,
    num_workers=0
):

    train_dataset = CTClassificationDataset(
        split="train"
    )

    val_dataset = CTClassificationDataset(
        split="validation"
    )

    test_dataset = CTClassificationDataset(
        split="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MEMBER 2 - CT DATASET TEST")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=2,
        num_workers=0
    )

    print()
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    images, targets, case_names = next(iter(train_loader))

    print()
    print("Batch image shape :", images.shape)
    print("Batch targets     :", targets)
    print("Case names        :", case_names)

    print("=" * 60)
    print("DATA LOADER TEST SUCCESSFUL")
    print("=" * 60)