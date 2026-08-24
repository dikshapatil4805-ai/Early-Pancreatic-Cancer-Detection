import os
import sys
import torch
from torch.utils.data import DataLoader


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

sys.path.append(
    os.path.join(BASE_DIR, "02_ct_model")
)


# ============================================================
# IMPORT DATASET AND MODEL
# ============================================================

from datasets.ct_segmentation_dataset import PancreasCTDataset
from models.unet3d import UNet3D


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DICE SCORE
# ============================================================

def dice_score(prediction, target, class_id):

    prediction = (prediction == class_id).float()
    target = (target == class_id).float()

    intersection = torch.sum(
        prediction * target
    )

    denominator = (
        torch.sum(prediction) +
        torch.sum(target)
    )

    # Avoid division by zero
    if denominator.item() == 0:
        return None

    dice = (
        2.0 * intersection
    ) / denominator

    return dice.item()


# ============================================================
# IOU SCORE
# ============================================================

def iou_score(prediction, target, class_id):

    prediction = (prediction == class_id)
    target = (target == class_id)

    intersection = torch.sum(
        prediction & target
    ).float()

    union = torch.sum(
        prediction | target
    ).float()

    # Avoid division by zero
    if union.item() == 0:
        return None

    iou = intersection / union

    return iou.item()


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("PANCREAS CT MODEL EVALUATION")
print("=" * 60)

print()
print("Device:", device)


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = PancreasCTDataset(
    split="test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)


print()
print("TEST DATASET")
print("-" * 40)

print(
    "Test cases:",
    len(test_dataset)
)


# ============================================================
# LOAD MODEL
# ============================================================

model = UNet3D(
    in_channels=1,
    num_classes=3
)

model = model.to(device)


# IMPORTANT:
# Load the NEW weighted model

model_path = os.path.join(
    BASE_DIR,
    "02_ct_model",
    "saved_models",
    "pancreas_unet3d_weighted.pth"
)


print()
print("LOADING TRAINED MODEL")
print("-" * 40)

print(model_path)


model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()

print("✓ Model loaded successfully")


# ============================================================
# SCORE STORAGE
# ============================================================

pancreas_dice_scores = []
tumor_dice_scores = []

pancreas_iou_scores = []
tumor_iou_scores = []

pancreas_cases = 0
tumor_cases = 0


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 60)
print("RUNNING TEST EVALUATION")
print("=" * 60)


with torch.no_grad():

    for batch_index, batch in enumerate(test_loader):

        # Dataset returns:
        # image, label

        images, labels = batch

        # Move data to device

        images = images.to(device)
        labels = labels.to(device)

        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        outputs = model(images)


        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        # ----------------------------------------------------
        # CHECK IF PANCREAS EXISTS IN GROUND TRUTH
        # ----------------------------------------------------

        pancreas_voxels = torch.sum(
            labels == 1
        ).item()


        if pancreas_voxels > 0:

            pancreas_cases += 1


            pancreas_dice = dice_score(
                predictions,
                labels,
                class_id=1
            )

            pancreas_iou = iou_score(
                predictions,
                labels,
                class_id=1
            )


            if pancreas_dice is not None:

                pancreas_dice_scores.append(
                    pancreas_dice
                )


            if pancreas_iou is not None:

                pancreas_iou_scores.append(
                    pancreas_iou
                )


        # ----------------------------------------------------
        # CHECK IF TUMOR EXISTS IN GROUND TRUTH
        # ----------------------------------------------------

        tumor_voxels = torch.sum(
            labels == 2
        ).item()


        if tumor_voxels > 0:

            tumor_cases += 1


            tumor_dice = dice_score(
                predictions,
                labels,
                class_id=2
            )

            tumor_iou = iou_score(
                predictions,
                labels,
                class_id=2
            )


            if tumor_dice is not None:

                tumor_dice_scores.append(
                    tumor_dice
                )


            if tumor_iou is not None:

                tumor_iou_scores.append(
                    tumor_iou
                )


        # ----------------------------------------------------
        # PRINT PROGRESS
        # ----------------------------------------------------

        if (batch_index + 1) % 5 == 0:

            print(
                f"Processed "
                f"{batch_index + 1}/"
                f"{len(test_loader)} cases"
            )


# ============================================================
# CALCULATE AVERAGES
# ============================================================

if len(pancreas_dice_scores) > 0:

    average_pancreas_dice = (
        sum(pancreas_dice_scores) /
        len(pancreas_dice_scores)
    )

else:

    average_pancreas_dice = 0.0


if len(pancreas_iou_scores) > 0:

    average_pancreas_iou = (
        sum(pancreas_iou_scores) /
        len(pancreas_iou_scores)
    )

else:

    average_pancreas_iou = 0.0


if len(tumor_dice_scores) > 0:

    average_tumor_dice = (
        sum(tumor_dice_scores) /
        len(tumor_dice_scores)
    )

else:

    average_tumor_dice = 0.0


if len(tumor_iou_scores) > 0:

    average_tumor_iou = (
        sum(tumor_iou_scores) /
        len(tumor_iou_scores)
    )

else:

    average_tumor_iou = 0.0


# ============================================================
# FINAL RESULTS
# ============================================================

print()

print("=" * 60)
print("FINAL EVALUATION RESULTS")
print("=" * 60)


print()

print("PANCREAS")
print("-" * 40)

print(
    "Test cases containing pancreas:",
    pancreas_cases
)

print(
    f"Average Dice Score: "
    f"{average_pancreas_dice:.4f}"
)

print(
    f"Average IoU Score: "
    f"{average_pancreas_iou:.4f}"
)


print()

print("TUMOR")
print("-" * 40)

print(
    "Test cases containing tumor:",
    tumor_cases
)

print(
    f"Average Dice Score: "
    f"{average_tumor_dice:.4f}"
)

print(
    f"Average IoU Score: "
    f"{average_tumor_iou:.4f}"
)


print()

print("=" * 60)
print("MODEL EVALUATION COMPLETED")
print("=" * 60)