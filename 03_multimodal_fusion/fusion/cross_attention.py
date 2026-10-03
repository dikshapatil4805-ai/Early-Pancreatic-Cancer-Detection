"""
Cross-attention multimodal fusion for PDAC risk prediction.

Inputs:
    CT feature vector:       [batch_size, 512]
    Clinical feature vector:[batch_size, 16]

Both modalities are projected into a shared embedding space.
Cross-attention then allows the representation of one modality
to attend to the other before risk classification.

Important:
This module defines and tests the architecture only.
It must not be trained using unverified CT/clinical pairings.
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion model.

    Parameters
    ----------
    ct_feature_dim : int
        Dimension of the CT feature vector.

    clinical_feature_dim : int
        Dimension of the clinical feature vector.

    embed_dim : int
        Shared embedding dimension.

    num_heads : int
        Number of attention heads.

    hidden_dim : int
        Hidden dimension of the final classifier.
    """

    def __init__(
        self,
        ct_feature_dim=512,
        clinical_feature_dim=16,
        embed_dim=64,
        num_heads=4,
        hidden_dim=128,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )

        self.ct_feature_dim = ct_feature_dim
        self.clinical_feature_dim = clinical_feature_dim
        self.embed_dim = embed_dim

        # Project each modality into the same embedding space.
        self.ct_projection = nn.Linear(
            ct_feature_dim,
            embed_dim
        )

        self.clinical_projection = nn.Linear(
            clinical_feature_dim,
            embed_dim
        )

        # CT queries clinical information.
        self.ct_to_clinical_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Clinical queries CT information.
        self.clinical_to_ct_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Final classifier receives both attended representations.
        self.classifier = nn.Sequential(
            nn.Linear(
                embed_dim * 2,
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
            Shape: [batch_size, 128]
            when embed_dim=64.
        """

        if ct_features.ndim != 2:
            raise ValueError(
                "ct_features must have shape "
                "[batch_size, ct_feature_dim]."
            )

        if clinical_features.ndim != 2:
            raise ValueError(
                "clinical_features must have shape "
                "[batch_size, clinical_feature_dim]."
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

        # [batch, 512] -> [batch, 64]
        ct_embedding = self.ct_projection(
            ct_features
        )

        # [batch, 16] -> [batch, 64]
        clinical_embedding = self.clinical_projection(
            clinical_features
        )

        # MultiheadAttention expects a sequence dimension.
        # Each modality is represented as one token:
        #
        # CT:       [batch, 1, 64]
        # Clinical: [batch, 1, 64]
        ct_token = ct_embedding.unsqueeze(1)
        clinical_token = clinical_embedding.unsqueeze(1)

        # CT attends to clinical information.
        ct_attended, _ = self.ct_to_clinical_attention(
            query=ct_token,
            key=clinical_token,
            value=clinical_token
        )

        # Clinical attends to CT information.
        clinical_attended, _ = self.clinical_to_ct_attention(
            query=clinical_token,
            key=ct_token,
            value=ct_token
        )

        # Remove the sequence dimension.
        ct_attended = ct_attended.squeeze(1)
        clinical_attended = clinical_attended.squeeze(1)

        # Combine both attended representations.
        fused_features = torch.cat(
            [
                ct_attended,
                clinical_attended
            ],
            dim=1
        )

        logits = self.classifier(
            fused_features
        ).squeeze(1)

        return logits, fused_features


if __name__ == "__main__":

    print("=" * 60)
    print("CROSS-ATTENTION FUSION TEST")
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

    model = CrossAttentionFusion()

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
    print("Expected fused dimension: 128")
    print("Actual fused dimension  :", fused_features.shape[1])

    assert fused_features.shape == (
        batch_size,
        128
    )

    assert logits.shape == (
        batch_size,
    )

    assert probabilities.shape == (
        batch_size,
    )

    print()
    print("Cross-attention fusion test passed.")
    print("=" * 60)