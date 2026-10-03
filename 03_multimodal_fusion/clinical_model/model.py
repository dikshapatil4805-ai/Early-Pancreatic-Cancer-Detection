"""
Clinical-only baseline model for pancreatic cancer detection.

This module:
- Loads and preprocesses clinical data using data_loader.py.
- Trains a small MLP classifier.
- Uses validation data for model selection.
- Evaluates the final model on the held-out test set.
- Reports accuracy, precision, recall, F1, ROC-AUC.
- Saves test metrics and a confusion matrix.
"""

from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from data_loader import prepare_clinical_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "03_multimodal_fusion" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODELS_DIR = PROJECT_ROOT / "03_multimodal_fusion" / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 8


class ClinicalMLP(nn.Module):
    """Small multilayer perceptron for clinical feature classification."""

    def __init__(self, input_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Make training as reproducible as practical."""

    np.random.seed(seed)
    torch.manual_seed(seed)


def dataframe_to_tensor(dataframe):
    """Convert a pandas DataFrame to a float32 PyTorch tensor."""

    return torch.tensor(
        dataframe.to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )


def evaluate_model(model, X, y):
    """Calculate classification metrics."""

    model.eval()

    with torch.no_grad():
        logits = model(X)
        probabilities = torch.sigmoid(logits).cpu().numpy()

    predictions = (probabilities >= 0.5).astype(int)
    y_true = y.cpu().numpy()

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
    }

    cm = confusion_matrix(y_true, predictions)

    return metrics, cm, probabilities, predictions


def train_model(model, X_train, y_train, X_validation, y_validation):
    """Train the MLP with early stopping using validation loss."""

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()

        permutation = torch.randperm(X_train.size(0))

        total_loss = 0.0

        for start in range(0, X_train.size(0), BATCH_SIZE):

            indices = permutation[start:start + BATCH_SIZE]

            batch_X = X_train[indices]
            batch_y = y_train[indices]

            optimizer.zero_grad()

            logits = model(batch_X)

            loss = criterion(logits, batch_y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item() * len(indices)

        train_loss = total_loss / X_train.size(0)

        model.eval()

        with torch.no_grad():
            validation_logits = model(X_validation)
            validation_loss = criterion(
                validation_logits,
                y_validation,
            ).item()

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def save_metrics(metrics, path: Path):
    """Save metrics to JSON."""

    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def save_confusion_matrix(cm, path: Path):
    """Save the confusion matrix as a PNG."""

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-PDAC", "PDAC"],
    )

    display.plot()

    plt.title("Clinical-Only MLP Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():

    set_seed()

    print("=" * 60)
    print("Clinical-Only Baseline Model")
    print("=" * 60)

    (
        preprocessor,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = prepare_clinical_data()

    print("\nInput feature dimensions:")
    print("Train:", X_train.shape)
    print("Validation:", X_validation.shape)
    print("Test:", X_test.shape)

    X_train_tensor = dataframe_to_tensor(X_train)
    X_validation_tensor = dataframe_to_tensor(X_validation)
    X_test_tensor = dataframe_to_tensor(X_test)

    y_train_tensor = torch.tensor(
        y_train.to_numpy(dtype=np.float32)
    )

    y_validation_tensor = torch.tensor(
        y_validation.to_numpy(dtype=np.float32)
    )

    y_test_tensor = torch.tensor(
        y_test.to_numpy(dtype=np.float32)
    )

    model = ClinicalMLP(
        input_dim=X_train.shape[1]
    )

    print("\nModel:")
    print(model)

    print("\nTraining...")
    model = train_model(
        model,
        X_train_tensor,
        y_train_tensor,
        X_validation_tensor,
        y_validation_tensor,
    )
    model_path = SAVED_MODELS_DIR / "clinical_mlp.pth"
    preprocessor_path = SAVED_MODELS_DIR / "clinical_preprocessor.joblib"

    torch.save(model.state_dict(), model_path)
    joblib.dump(preprocessor, preprocessor_path)

    print("\nSaved model artifacts:")
    print(model_path)
    print(preprocessor_path)

    validation_metrics, _, _, _ = evaluate_model(
        model,
        X_validation_tensor,
        y_validation_tensor,
    )

    test_metrics, test_cm, _, _ = evaluate_model(
        model,
        X_test_tensor,
        y_test_tensor,
    )

    print("\nValidation Metrics")
    print("-" * 40)

    for name, value in validation_metrics.items():
        print(f"{name:10s}: {value:.4f}")

    print("\nTest Metrics")
    print("-" * 40)

    for name, value in test_metrics.items():
        print(f"{name:10s}: {value:.4f}")

    metrics_output = {
        "model": "clinical_only_mlp",
        "random_seed": RANDOM_SEED,
        "input_features": int(X_train.shape[1]),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_validation)),
        "test_samples": int(len(X_test)),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_confusion_matrix": test_cm.tolist(),
    }

    metrics_path = RESULTS_DIR / "clinical_metrics.json"
    confusion_matrix_path = RESULTS_DIR / "clinical_confusion_matrix.png"

    save_metrics(
        metrics_output,
        metrics_path,
    )

    save_confusion_matrix(
        test_cm,
        confusion_matrix_path,
    )

    print("\nSaved results:")
    print(metrics_path)
    print(confusion_matrix_path)

    print("\nClinical baseline completed successfully.")


if __name__ == "__main__":
    main()