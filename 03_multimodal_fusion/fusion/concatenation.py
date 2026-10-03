"""
Multimodal feature concatenation for PDAC risk prediction.

CT features:
    512-dimensional vector from the 3D ResNet CT encoder.

Clinical features:
    16-dimensional representation from the clinical MLP.

The two feature vectors are concatenated and passed through
a small neural network for binary PDAC risk prediction.

Important:
This module defines the fusion architecture only. It does not
assume that PANORAMA clinical records are paired with Task07
Pancreas CT cases. Valid paired data must be independently
verified before training this model.
"""

import torch
import torch.nn as nn


class ConcatenationFusion(nn.Module):
    """
    Concatenation-based multimodal fusion model.

    Parameters
    ----------
    ct_feature_dim : int
        Dimension of the CT feature vector.

    clinical_feature_dim : int
        Dimension of the clinical feature vector.

    hidden_dim : int
        Number of units in the fusion layer.
    """

    def __init__(
        self,
        ct_feature_dim=512,
        clinical_feature_dim=16,
        hidden_dim=128,
    ):
        super().__init__()

        self.ct_feature_dim = ct_feature_dim
        self.clinical_feature_dim = clinical_feature_dim
        self.fused_dim = (
            ct_feature_dim + clinical_feature_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                self.fused_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(
        self,
        ct_features,
        clinical_features
    ):
        """
        Forward pass.

        Parameters
        ----------
        ct_features : torch.Tensor
            Shape: [batch_size, 512]

        clinical_features : torch.Tensor
            Shape: [batch_size, 16]

        Returns
        -------
        logits : torch.Tensor
            Shape: [batch_size]

        fused_features : torch.Tensor
            Shape: [batch_size, 528]
        """

        if ct_features.ndim != 2:
            raise ValueError(
                "ct_features must have shape "
                "[batch_size, ct_feature_dim]"
            )

        if clinical_features.ndim != 2:
            raise ValueError(
                "clinical_features must have shape "
                "[batch_size, clinical_feature_dim]"
            )

        if ct_features.size(0) != clinical_features.size(0):
            raise ValueError(
                "CT and clinical batches must contain "
                "the same number of samples."
            )

        if ct_features.size(1) != self.ct_feature_dim:
            raise ValueError(
                f"Expected CT feature dimension "
                f"{self.ct_feature_dim}, got "
                f"{ct_features.size(1)}."
            )

        if clinical_features.size(1) != self.clinical_feature_dim:
            raise ValueError(
                f"Expected clinical feature dimension "
                f"{self.clinical_feature_dim}, got "
                f"{clinical_features.size(1)}."
            )

        fused_features = torch.cat(
            [
                ct_features,
                clinical_features
            ],
            dim=1
        )

        logits = self.fusion(
            fused_features
        ).squeeze(1)

        return logits, fused_features


if __name__ == "__main__":

    print("=" * 60)
    print("MULTIMODAL CONCATENATION FUSION TEST")
    print("=" * 60)

    batch_size = 4

    # Dummy feature vectors for architecture testing only.
    ct_features = torch.randn(
        batch_size,
        512
    )

    clinical_features = torch.randn(
        batch_size,
        16
    )

    model = ConcatenationFusion()

    logits, fused_features = model(
        ct_features,
        clinical_features
    )

    probabilities = torch.sigmoid(
        logits
    )

    print()
    print("CT feature shape       :", ct_features.shape)
    print(
        "Clinical feature shape :",
        clinical_features.shape
    )
    print(
        "Fused feature shape    :",
        fused_features.shape
    )
    print(
        "Logits shape           :",
        logits.shape
    )
    print(
        "Probability shape      :",
        probabilities.shape
    )

    print()
    print("Expected fused dimension: 528")
    print("Actual fused dimension  :", model.fused_dim)

    assert fused_features.shape == (
        batch_size,
        528
    )

    assert logits.shape == (
        batch_size,
    )

    assert probabilities.shape == (
        batch_size,
    )

    print()
    print("Concatenation fusion test passed.")
    print("=" * 60)