import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
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


print("=" * 60)
print("PANCREAS CT MODEL TRAINING")
print("=" * 60)

print()
print("Device:", device)


# ============================================================
# DATASETS
# ============================================================

train_dataset = PancreasCTDataset(
    split="train"
)

validation_dataset = PancreasCTDataset(
    split="validation"
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=1,
    shuffle=False
)


print()
print("DATASET INFORMATION")
print("-" * 40)

print(
    "Training cases:",
    len(train_dataset)
)

print(
    "Validation cases:",
    len(validation_dataset)
)


# ============================================================
# MODEL
# ============================================================

model = UNet3D(
    in_channels=1,
    num_classes=3
)

model = model.to(device)


# ============================================================
# CLASS WEIGHTS
# ============================================================

# Class 0 = Background
# Class 1 = Pancreas
# Class 2 = Tumor

class_weights = torch.tensor(
    [0.01, 1.0, 5.0],
    dtype=torch.float32
).to(device)


print()
print("CLASS WEIGHTS")
print("-" * 40)
print("Background:", class_weights[0].item())
print("Pancreas:", class_weights[1].item())
print("Tumor:", class_weights[2].item())


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

NUM_EPOCHS = 10


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(NUM_EPOCHS):

    print()
    print("=" * 60)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print("=" * 60)


    # ========================================================
    # TRAINING MODE
    # ========================================================

    model.train()

    running_loss = 0.0


    # ========================================================
    # TRAINING BATCHES
    # ========================================================

    for batch_index, batch in enumerate(train_loader):

        # Dataset returns:
        # image, label

        images, labels = batch


        # Move data to device

        images = images.to(device)

        labels = labels.to(device)


        # Ensure integer labels

        labels = labels.long()


        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        outputs = model(images)


        # ----------------------------------------------------
        # CALCULATE LOSS
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # SAVE LOSS
        # ----------------------------------------------------

        running_loss += loss.item()


        # ----------------------------------------------------
        # PRINT PROGRESS
        # ----------------------------------------------------

        if (batch_index + 1) % 10 == 0:

            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)} "
                f"| Loss: "
                f"{loss.item():.4f}"
            )


    # ========================================================
    # AVERAGE TRAINING LOSS
    # ========================================================

    average_train_loss = (
        running_loss /
        len(train_loader)
    )


    print()

    print(
        f"Average Training Loss: "
        f"{average_train_loss:.4f}"
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    validation_loss = 0.0


    with torch.no_grad():

        for batch in validation_loader:

            images, labels = batch


            # Move data to device

            images = images.to(device)

            labels = labels.to(device)


            # Ensure integer labels

            labels = labels.long()


            # Forward pass

            outputs = model(images)


            # Calculate validation loss

            loss = criterion(
                outputs,
                labels
            )


            validation_loss += loss.item()


    # ========================================================
    # AVERAGE VALIDATION LOSS
    # ========================================================

    average_validation_loss = (
        validation_loss /
        len(validation_loader)
    )


    print(
        f"Average Validation Loss: "
        f"{average_validation_loss:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_dir = os.path.join(
    BASE_DIR,
    "02_ct_model",
    "saved_models"
)

os.makedirs(
    model_dir,
    exist_ok=True
)


# Save new weighted model separately

model_path = os.path.join(
    model_dir,
    "pancreas_unet3d_weighted.pth"
)


torch.save(
    model.state_dict(),
    model_path
)


# ============================================================
# TRAINING COMPLETED
# ============================================================

print()

print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print()

print("Model saved to:")

print(model_path)