import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# PANCREAS CT DATASET
# ============================================================

class PancreasCTDataset(Dataset):

    def __init__(self, split="train"):

        # Project root directory
        BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )
        )

        # ====================================================
        # RESIZED CT IMAGES
        # ====================================================

        self.image_dir = os.path.join(
            BASE_DIR,
            "data",
            "processed",
            "ct",
            "resized_images"
        )

        # ====================================================
        # RESIZED LABELS
        # ====================================================

        self.label_dir = os.path.join(
            BASE_DIR,
            "data",
            "processed",
            "ct",
            "resized_labels"
        )

        # ====================================================
        # DATASET SPLIT FILE
        # ====================================================

        split_file = os.path.join(
            BASE_DIR,
            "data",
            "processed",
            "ct",
            "splits",
            split + ".txt"
        )

        # Load case names
        with open(split_file, "r") as f:

            self.cases = [
                line.strip()
                for line in f.readlines()
            ]

    # ========================================================
    # DATASET LENGTH
    # ========================================================

    def __len__(self):

        return len(self.cases)

    # ========================================================
    # GET ONE CASE
    # ========================================================

    def __getitem__(self, index):

        case_name = self.cases[index]

        # Image path
        image_path = os.path.join(
            self.image_dir,
            case_name
        )

        # Label path
        label_path = os.path.join(
            self.label_dir,
            case_name
        )

        # Load CT image
        image = nib.load(
            image_path
        ).get_fdata()

        # Load segmentation label
        label = nib.load(
            label_path
        ).get_fdata()

        # Convert image to float32
        image = np.array(
            image,
            dtype=np.float32
        )

        # Convert label to int64
        label = np.array(
            label,
            dtype=np.int64
        )

        # Add channel dimension
        # From: (64, 64, 64)
        # To:   (1, 64, 64, 64)

        image = np.expand_dims(
            image,
            axis=0
        )

        # Convert to PyTorch tensors

        image = torch.from_numpy(
            image
        )

        label = torch.from_numpy(
            label
        )

        return image, label


# ============================================================
# TEST DATASET
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PANCREAS CT PYTORCH DATASET")
    print("=" * 60)

    # Load training dataset

    dataset = PancreasCTDataset(
        split="train"
    )

    print()

    print(
        f"Training cases: {len(dataset)}"
    )

    print()

    # Load first sample

    image, label = dataset[0]

    print("SAMPLE INFORMATION")
    print("-" * 40)

    print(
        f"CT tensor shape: {image.shape}"
    )

    print(
        f"CT data type: {image.dtype}"
    )

    print(
        f"CT minimum: {image.min().item()}"
    )

    print(
        f"CT maximum: {image.max().item()}"
    )

    print()

    print(
        f"Label tensor shape: {label.shape}"
    )

    print(
        f"Label data type: {label.dtype}"
    )

    print(
        f"Unique labels: {torch.unique(label)}"
    )

    print()

    print("=" * 60)
    print("PYTORCH DATASET TEST COMPLETED")
    print("=" * 60)