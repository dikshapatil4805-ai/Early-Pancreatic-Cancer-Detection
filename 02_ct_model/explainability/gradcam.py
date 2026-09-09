import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


# =========================================================
# PATH SETUP
# =========================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "best_resnet3d.pth"
)

OUTPUT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "gradcam_result.png"
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
# LOAD CT
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

    if volume.shape != (64, 64, 64):

        raise ValueError(
            f"Unexpected CT shape: {volume.shape}\n"
            "Expected: (64, 64, 64)"
        )

    # Normalize
    volume_min = volume.min()
    volume_max = volume.max()

    if volume_max > volume_min:

        volume = (
            volume - volume_min
        ) / (
            volume_max - volume_min
        )

    tensor = torch.from_numpy(
        volume
    )

    # [D,H,W]
    # →
    # [1,1,D,H,W]

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    return tensor


# =========================================================
# GRAD-CAM CLASS
# =========================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_hook = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )

    # -----------------------------------------------------

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    # -----------------------------------------------------

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    # -----------------------------------------------------

    def remove_hooks(self):

        self.forward_hook.remove()
        self.backward_hook.remove()

    # -----------------------------------------------------

    def generate(
        self,
        input_tensor,
        target_class=None
    ):

        self.model.zero_grad()

        logits, features = self.model(
            input_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        if target_class is None:

            target_class = predicted_class

        score = logits[
            0,
            target_class
        ]

        score.backward()

        if (
            self.activations is None
            or self.gradients is None
        ):

            raise RuntimeError(
                "Grad-CAM activations or gradients "
                "were not captured."
            )

        activations = self.activations
        gradients = self.gradients

        # -------------------------------------------------
        # Global average pooling of gradients
        # -------------------------------------------------

        weights = torch.mean(
            gradients,
            dim=(2, 3, 4),
            keepdim=True
        )

        # -------------------------------------------------
        # Weighted activation maps
        # -------------------------------------------------

        cam = torch.sum(
            weights * activations,
            dim=1
        )

        cam = F.relu(cam)

        # -------------------------------------------------
        # Resize CAM to original CT size
        # -------------------------------------------------

        cam = cam.unsqueeze(1)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[
                2:
            ],
            mode="trilinear",
            align_corners=False
        )

        cam = cam.squeeze()

        # -------------------------------------------------
        # Normalize CAM
        # -------------------------------------------------

        cam = cam.detach().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(
                cam
            )

        return (
            cam,
            predicted_class,
            probabilities.detach().cpu().numpy()[0]
        )


# =========================================================
# CREATE VISUALIZATION
# =========================================================

def save_gradcam_image(
    ct_volume,
    heatmap,
    predicted_class,
    probabilities
):

    # -----------------------------------------------------
    # Select central slice
    # -----------------------------------------------------

    depth = ct_volume.shape[0]

    slice_index = depth // 2

    ct_slice = ct_volume[
        slice_index
    ]

    heatmap_slice = heatmap[
        slice_index
    ]

    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------

    if predicted_class == 0:

        prediction_label = "No Tumor"

    else:

        prediction_label = "Tumor"

    probability = probabilities[
        predicted_class
    ]

    # -----------------------------------------------------
    # Figure
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # -----------------------------------------------------
    # Original CT
    # -----------------------------------------------------

    axes[0].imshow(
        ct_slice,
        cmap="gray"
    )

    axes[0].set_title(
        "Original CT Slice"
    )

    axes[0].axis("off")

    # -----------------------------------------------------
    # Heatmap
    # -----------------------------------------------------

    axes[1].imshow(
        heatmap_slice,
        cmap="jet"
    )

    axes[1].set_title(
        "Grad-CAM Heatmap"
    )

    axes[1].axis("off")

    # -----------------------------------------------------
    # Overlay
    # -----------------------------------------------------

    axes[2].imshow(
        ct_slice,
        cmap="gray"
    )

    axes[2].imshow(
        heatmap_slice,
        cmap="jet",
        alpha=0.45
    )

    axes[2].set_title(
        "CT + Grad-CAM"
    )

    axes[2].axis("off")

    # -----------------------------------------------------
    # Main title
    # -----------------------------------------------------

    fig.suptitle(
        f"Prediction: {prediction_label} | "
        f"Probability: {probability:.4f}",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# MAIN
# =========================================================

def generate_gradcam():

    print("=" * 60)
    print("MEMBER 2 - CT MODEL GRAD-CAM")
    print("=" * 60)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model, device = load_model()

    print("\nDevice:", device)

    # -----------------------------------------------------
    # Select target layer
    # -----------------------------------------------------

    # Last convolutional layer of ResNet3D
    target_layer = model.layer4[0].conv2

    # -----------------------------------------------------
    # Find CT
    # -----------------------------------------------------

    ct_dir = os.path.join(
        os.path.dirname(PROJECT_DIR),
        "data",
        "processed",
        "ct",
        "resized_images"
    )

    if not os.path.exists(ct_dir):

        raise FileNotFoundError(
            f"CT directory not found:\n{ct_dir}"
        )

    ct_files = sorted(
        [
            f
            for f in os.listdir(ct_dir)
            if f.endswith(".nii.gz")
        ]
    )

    if len(ct_files) == 0:

        raise FileNotFoundError(
            "No CT files found."
        )

    ct_file = ct_files[0]

    ct_path = os.path.join(
        ct_dir,
        ct_file
    )

    print("\nInput CT:")
    print(ct_path)

    # -----------------------------------------------------
    # Load CT
    # -----------------------------------------------------

    input_tensor = load_ct_volume(
        ct_path
    )

    input_tensor = input_tensor.to(
        device
    )

    # -----------------------------------------------------
    # Grad-CAM
    # -----------------------------------------------------

    gradcam = GradCAM(
        model,
        target_layer
    )

    heatmap, predicted_class, probabilities = (
        gradcam.generate(
            input_tensor
        )
    )

    gradcam.remove_hooks()

    # -----------------------------------------------------
    # Original volume
    # -----------------------------------------------------

    ct_volume = (
        input_tensor
        .detach()
        .cpu()
        .numpy()[0, 0]
    )

    # -----------------------------------------------------
    # Save image
    # -----------------------------------------------------

    save_gradcam_image(
        ct_volume,
        heatmap,
        predicted_class,
        probabilities
    )

    # -----------------------------------------------------
    # Print result
    # -----------------------------------------------------

    if predicted_class == 0:

        prediction_label = "No Tumor"

    else:

        prediction_label = "Tumor"

    print("\nPrediction:")
    print(prediction_label)

    print("\nClass:")
    print(predicted_class)

    print("\nClass Probabilities:")

    print(
        f"No Tumor : {probabilities[0]:.4f}"
    )

    print(
        f"Tumor    : {probabilities[1]:.4f}"
    )

    print("\nGrad-CAM image saved at:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("GRAD-CAM TEST SUCCESSFUL")
    print("=" * 60)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    generate_gradcam()