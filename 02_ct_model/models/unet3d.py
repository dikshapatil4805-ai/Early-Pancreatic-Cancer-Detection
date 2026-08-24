import torch
import torch.nn as nn


# ============================================================
# DOUBLE CONVOLUTION BLOCK
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)


# ============================================================
# 3D U-NET
# ============================================================

class UNet3D(nn.Module):

    def __init__(
        self,
        in_channels=1,
        num_classes=3
    ):

        super(UNet3D, self).__init__()

        # Encoder
        self.encoder1 = DoubleConv(
            in_channels,
            16
        )

        self.pool1 = nn.MaxPool3d(
            kernel_size=2
        )

        self.encoder2 = DoubleConv(
            16,
            32
        )

        self.pool2 = nn.MaxPool3d(
            kernel_size=2
        )

        self.encoder3 = DoubleConv(
            32,
            64
        )

        self.pool3 = nn.MaxPool3d(
            kernel_size=2
        )

        # Bottleneck
        self.bottleneck = DoubleConv(
            64,
            128
        )

        # Decoder
        self.upconv3 = nn.ConvTranspose3d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.decoder3 = DoubleConv(
            128,
            64
        )

        self.upconv2 = nn.ConvTranspose3d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.decoder2 = DoubleConv(
            64,
            32
        )

        self.upconv1 = nn.ConvTranspose3d(
            32,
            16,
            kernel_size=2,
            stride=2
        )

        self.decoder1 = DoubleConv(
            32,
            16
        )

        # Final classification layer
        self.final_conv = nn.Conv3d(
            16,
            num_classes,
            kernel_size=1
        )


    def forward(self, x):

        # Encoder
        enc1 = self.encoder1(x)

        enc2 = self.encoder2(
            self.pool1(enc1)
        )

        enc3 = self.encoder3(
            self.pool2(enc2)
        )

        # Bottleneck
        bottleneck = self.bottleneck(
            self.pool3(enc3)
        )

        # Decoder

        dec3 = self.upconv3(
            bottleneck
        )

        dec3 = torch.cat(
            [dec3, enc3],
            dim=1
        )

        dec3 = self.decoder3(
            dec3
        )


        dec2 = self.upconv2(
            dec3
        )

        dec2 = torch.cat(
            [dec2, enc2],
            dim=1
        )

        dec2 = self.decoder2(
            dec2
        )


        dec1 = self.upconv1(
            dec2
        )

        dec1 = torch.cat(
            [dec1, enc1],
            dim=1
        )

        dec1 = self.decoder1(
            dec1
        )

        # Final output
        output = self.final_conv(
            dec1
        )

        return output


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("3D U-NET MODEL TEST")
    print("=" * 60)

    # Create model
    model = UNet3D(
        in_channels=1,
        num_classes=3
    )

    print()
    print("MODEL CREATED SUCCESSFULLY")

    print()

    # Dummy CT volume
    x = torch.randn(
        1,
        1,
        128,
        128,
        128
    )

    print(
        f"Input shape: {x.shape}"
    )

    print()

    # Forward pass
    with torch.no_grad():

        output = model(x)

    print(
        f"Output shape: {output.shape}"
    )

    print()

    print(
        "Expected output shape: "
        "[1, 3, 128, 128, 128]"
    )

    print()

    # Count parameters
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print()
    print("=" * 60)
    print("3D U-NET MODEL TEST COMPLETED")
    print("=" * 60)