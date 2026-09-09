import os
import sys
import torch
import numpy as np
import nibabel as nib


# =========================================================
# PATH SETUP
# =========================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(0, PROJECT_DIR)


# =========================================================
# IMPORT MODEL
# =========================================================

from models.resnet_model import ResNet3DClassifier


# =========================================================
# CONFIGURATION
# =========================================================

NUM_CLASSES = 2
FEATURE_DIM = 512

CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "best_resnet3d.pth"
)


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ResNet3DClassifier(
        num_classes=NUM_CLASSES
    )

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["state_dict"]
            )

        else:
            model.load_state_dict(checkpoint)

    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, device


# =========================================================
# LOAD CT VOLUME
# =========================================================

def load_ct_volume(ct_path):

    if not os.path.exists(ct_path):
        raise FileNotFoundError(
            f"CT file not found:\n{ct_path}"
        )

    nii = nib.load(ct_path)

    volume = nii.get_fdata().astype(
        np.float32
    )

    # Expected resized volume:
    # 64 x 64 x 64

    if volume.shape != (64, 64, 64):

        raise ValueError(
            f"Unexpected CT shape: {volume.shape}\n"
            f"Expected: (64, 64, 64)"
        )

    # Ensure values are in approximately [0, 1]
    volume_min = volume.min()
    volume_max = volume.max()

    if volume_max > volume_min:

        volume = (
            volume - volume_min
        ) / (
            volume_max - volume_min
        )

    # Convert:
    # [D, H, W]
    # to
    # [1, 1, D, H, W]

    volume = torch.from_numpy(volume)

    volume = volume.unsqueeze(0)
    volume = volume.unsqueeze(0)

    return volume


# =========================================================
# CT INFERENCE
# =========================================================

def predict_ct(ct_path):

    model, device = load_model()

    ct_tensor = load_ct_volume(
        ct_path
    )

    ct_tensor = ct_tensor.to(device)

    with torch.no_grad():

        logits, features = model(
            ct_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

    # Probability of predicted class
    predicted_probability = probabilities[
        0,
        prediction
    ].item()

    # 512-dimensional CT feature vector
    feature_vector = features[
        0
    ].cpu().numpy()

    # -----------------------------------------------------
    # Convert prediction to label
    # -----------------------------------------------------

    if prediction == 0:
        prediction_label = "No Tumor"

    else:
        prediction_label = "Tumor"

    result = {

        "prediction": prediction_label,

        "prediction_class": prediction,

        "probability": float(
            predicted_probability
        ),

        "feature_vector": feature_vector.tolist()
    }

    return result


# =========================================================
# TEST INFERENCE
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MEMBER 2 - CT MODEL INFERENCE")
    print("=" * 60)

    # -----------------------------------------------------
    # Automatically select one test CT
    # -----------------------------------------------------

    test_images_dir = os.path.join(
        os.path.dirname(PROJECT_DIR),
        "data",
        "processed",
        "ct",
        "resized_images"
    )

    if not os.path.exists(test_images_dir):

        raise FileNotFoundError(
            f"\nResized CT folder not found:\n"
            f"{test_images_dir}"
        )

    ct_files = [
        f for f in os.listdir(test_images_dir)
        if f.endswith(".nii.gz")
    ]

    if len(ct_files) == 0:

        raise FileNotFoundError(
            "\nNo CT files found."
        )

    ct_file = sorted(ct_files)[0]

    ct_path = os.path.join(
        test_images_dir,
        ct_file
    )

    print("\nInput CT:")
    print(ct_path)

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    result = predict_ct(
        ct_path
    )

    print("\nPrediction:")
    print(
        result["prediction"]
    )

    print("\nPrediction Class:")
    print(
        result["prediction_class"]
    )

    print("\nProbability:")
    print(
        f'{result["probability"]:.4f}'
    )

    print("\nFeature Vector:")
    print(
        "Dimension:",
        len(result["feature_vector"])
    )

    print(
        "First 10 values:",
        result["feature_vector"][:10]
    )

    print("\n" + "=" * 60)
    print("INFERENCE TEST SUCCESSFUL")
    print("=" * 60)