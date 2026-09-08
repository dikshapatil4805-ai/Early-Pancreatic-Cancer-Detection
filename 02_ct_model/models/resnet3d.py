import torch
import torch.nn as nn


class BasicBlock3D(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):

        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm3d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm3d(
            out_channels
        )

        self.shortcut = nn.Sequential()

        if (
            stride != 1
            or in_channels != out_channels
        ):

            self.shortcut = nn.Sequential(

                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm3d(
                    out_channels
                )
            )

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


class ResNet3D(nn.Module):

    def __init__(
        self,
        num_classes=2,
        feature_dim=128
    ):

        super().__init__()

        self.in_channels = 32

        self.conv1 = nn.Conv3d(
            1,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm3d(32)

        self.relu = nn.ReLU(
            inplace=True
        )

        self.layer1 = self._make_layer(
            32,
            2,
            stride=1
        )

        self.layer2 = self._make_layer(
            64,
            2,
            stride=2
        )

        self.layer3 = self._make_layer(
            128,
            2,
            stride=2
        )

        self.layer4 = self._make_layer(
            256,
            2,
            stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool3d(
            (1, 1, 1)
        )

        self.feature_layer = nn.Sequential(

            nn.Linear(
                256,
                feature_dim
            ),

            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(
            feature_dim,
            num_classes
        )

    def _make_layer(
        self,
        out_channels,
        blocks,
        stride
    ):

        layers = []

        layers.append(
            BasicBlock3D(
                self.in_channels,
                out_channels,
                stride
            )
        )

        self.in_channels = out_channels

        for _ in range(1, blocks):

            layers.append(
                BasicBlock3D(
                    out_channels,
                    out_channels
                )
            )

        return nn.Sequential(*layers)

    def forward(
        self,
        x,
        return_features=False
    ):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Important for Grad-CAM
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            1
        )

        features = self.feature_layer(x)

        logits = self.classifier(
            features
        )

        if return_features:
            return logits, features

        return logits