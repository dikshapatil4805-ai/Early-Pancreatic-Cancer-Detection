import torch
import torch.nn as nn


# ---------------------------------------------------------
# 3D RESNET BASIC BLOCK
# ---------------------------------------------------------
class BasicBlock3D(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm3d(out_channels)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# ---------------------------------------------------------
# 3D RESNET MODEL
# ---------------------------------------------------------
class ResNet3DClassifier(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        # Initial layer
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # ResNet blocks
        self.layer1 = self._make_layer(
            64, 64, blocks=1, stride=1
        )

        self.layer2 = self._make_layer(
            64, 128, blocks=1, stride=2
        )

        self.layer3 = self._make_layer(
            128, 256, blocks=1, stride=2
        )

        self.layer4 = self._make_layer(
            256, 512, blocks=1, stride=2
        )

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Feature vector size = 512
        self.feature_dim = 512

        # Final classifier
        self.fc = nn.Linear(
            self.feature_dim,
            num_classes
        )

    def _make_layer(
        self,
        in_channels,
        out_channels,
        blocks,
        stride
    ):

        layers = []

        layers.append(
            BasicBlock3D(
                in_channels,
                out_channels,
                stride
            )
        )

        for _ in range(1, blocks):
            layers.append(
                BasicBlock3D(
                    out_channels,
                    out_channels
                )
            )

        return nn.Sequential(*layers)

    def forward_features(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        # [batch, 512, 1, 1, 1]
        x = torch.flatten(x, 1)

        # [batch, 512]
        return x

    def forward(self, x):

        features = self.forward_features(x)

        logits = self.fc(features)

        return logits, features


# ---------------------------------------------------------
# TEST MODEL
# ---------------------------------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("MEMBER 2 - 3D RESNET MODEL TEST")
    print("=" * 60)

    # Example CT batch
    # [Batch, Channel, Depth, Height, Width]
    x = torch.randn(2, 1, 64, 64, 64)

    # Create model
    model = ResNet3DClassifier(num_classes=2)

    # Forward pass
    logits, features = model(x)

    print("Input shape       :", x.shape)
    print("Logits shape      :", logits.shape)
    print("Feature shape     :", features.shape)

    # Probability
    probabilities = torch.softmax(logits, dim=1)

    print("Probability shape :", probabilities.shape)

    print("=" * 60)
    print("MODEL TEST SUCCESSFUL")
    print("=" * 60)