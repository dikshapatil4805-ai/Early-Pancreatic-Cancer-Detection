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


from datasets.ct_segmentation_dataset import PancreasCTDataset
from models.unet3d import UNet3D


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 60)
print("DEBUGGING MODEL PREDICTIONS")
print("=" * 60)

print()
print("Device:", device)


# ============================================================
# LOAD TEST DATASET
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
print("Test cases:", len(test_dataset))


# ============================================================
# LOAD MODEL
# ============================================================

model = UNet3D(
    in_channels=1,
    num_classes=3
)

model = model.to(device)


model_path = os.path.join(
    BASE_DIR,
    "02_ct_model",
    "saved_models",
    "pancreas_unet3d_weighted.pth"
)


model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()


print()
print("✓ Model loaded successfully")


# ============================================================
# TEST FIRST 5 CASES
# ============================================================

print()
print("=" * 60)
print("PREDICTION CHECK")
print("=" * 60)


with torch.no_grad():

    for batch_index, batch in enumerate(test_loader):

        images = batch[0].to(device)
        labels = batch[1].to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        print()
        print(f"CASE {batch_index + 1}")

        print("-" * 40)

        print(
            "Ground truth labels:",
            torch.unique(labels).cpu().numpy()
        )

        print(
            "Predicted labels:",
            torch.unique(predictions).cpu().numpy()
        )

        # Count voxels for each class
        for class_id in [0, 1, 2]:

            ground_truth_count = torch.sum(
                labels == class_id
            ).item()

            prediction_count = torch.sum(
                predictions == class_id
            ).item()

            print()

            print(
                f"Class {class_id}"
            )

            print(
                f"Ground truth voxels: "
                f"{ground_truth_count}"
            )

            print(
                f"Predicted voxels: "
                f"{prediction_count}"
            )

        # Only check first 5 cases
        if batch_index == 4:
            break


print()

print("=" * 60)
print("DEBUGGING COMPLETED")
print("=" * 60)