import os
import sys

 feature/ct-model
# Add 02_ct_model folder to Python path
PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_DIR)

import random
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import create_dataloaders
from models.resnet_model import ResNet3DClassifier
from validate import validate_model

# ---------------------------------------------------------
# RANDOM SEED
# ---------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# LOAD CONFIG
# ---------------------------------------------------------
def load_config():

    config_path = os.path.join(
        os.path.dirname(__file__),
        "config.yaml"
    )

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------
def train():

    print("=" * 60)
    print("MEMBER 2 - CT RESNET TRAINING")
    print("=" * 60)

    # Load configuration
    config = load_config()

    # Set seed
    set_seed(config["seed"])

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # -----------------------------------------------------
    # DATA LOADERS
    # -----------------------------------------------------

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"]
    )

    print("Training cases  :", len(train_loader.dataset))
    print("Validation cases:", len(val_loader.dataset))
    print("Test cases      :", len(test_loader.dataset))

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = ResNet3DClassifier(
        num_classes=config["model"]["num_classes"]
    )

    model = model.to(device)

    print("Model: 3D ResNet")

    # -----------------------------------------------------
    # LOSS
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # OPTIMIZER
    # -----------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    # -----------------------------------------------------
    # LEARNING RATE SCHEDULER
    # -----------------------------------------------------

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # -----------------------------------------------------
    # CHECKPOINT DIRECTORY
    # -----------------------------------------------------

    checkpoint_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "checkpoints"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    best_model_path = os.path.join(
        checkpoint_dir,
        "best_resnet3d.pth"
    )

    # -----------------------------------------------------
    # TRAINING VARIABLES
    # -----------------------------------------------------

    epochs = config["training"]["epochs"]

    patience = config["training"]["patience"]

    best_val_loss = float("inf")

    epochs_without_improvement = 0

    # -----------------------------------------------------
    # TRAINING LOOP
    # -----------------------------------------------------

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets, _ in train_loader:

            images = images.to(device)
            targets = targets.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            logits, features = model(images)

            # Calculate loss
            loss = criterion(
                logits,
                targets
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

        # Training results
        train_loss = running_loss / total
        train_accuracy = correct / total

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        val_loss, val_accuracy = validate_model(
            model,
            val_loader,
            criterion,
            device
        )

        # Scheduler
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # -------------------------------------------------
        # PRINT RESULTS
        # -------------------------------------------------

        print(
            f"\nEpoch [{epoch + 1}/{epochs}]"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        print(
            f"Learning Rate: {current_lr:.6f}"
        )

        # -------------------------------------------------
        # SAVE BEST MODEL
        # -------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                },
                best_model_path
            )

            print(
                "✓ Best model saved!"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement: "
                f"{epochs_without_improvement}/{patience}"
            )

        # -------------------------------------------------
        # EARLY STOPPING
        # -------------------------------------------------

        if epochs_without_improvement >= patience:

            print(
                "\nEarly stopping triggered."
            )

            break

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        "Best model saved at:"
    )

    print(
        best_model_path
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    train()

