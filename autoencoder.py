import torch
import torch.nn as nn

from constant import IMG_SIZE

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()


        self.enc1_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc1_relu = nn.ReLU()
        self.enc1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc2_relu = nn.ReLU()
        self.enc2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3_conv = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc3_relu = nn.ReLU()
        self.enc3_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4_conv = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.enc4_relu = nn.ReLU()
        self.enc4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck_conv = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bottleneck_relu = nn.ReLU()

        self.up4_transpose = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up4_relu = nn.ReLU()

        self.up3_transpose = nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2)
        self.up3_relu = nn.ReLU()


        self.up2_transpose = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2)
        self.up2_relu = nn.ReLU()


        self.up1_transpose = nn.ConvTranspose2d(256, out_channels, kernel_size=2, stride=2)
        self.out_activation = nn.Sigmoid()

    def forward(self, x):
        skip1 = self.enc1_relu(self.enc1_conv(x))
        x1 = self.enc1_pool(skip1)

        skip2 = self.enc2_relu(self.enc2_conv(x1))
        x2 = self.enc2_pool(skip2)

        skip3 = self.enc3_relu(self.enc3_conv(x2))
        x3 = self.enc3_pool(skip3)

        skip4 = self.enc4_relu(self.enc4_conv(x3))
        x4 = self.enc4_pool(skip4)

        bn = self.bottleneck_relu(self.bottleneck_conv(x4))

        d4 = self.up4_transpose(bn)
        d4 = self.up4_relu(d4)
        d4 = torch.cat([d4, skip4], dim=1)

        d3 = self.up3_transpose(d4)
        d3 = self.up3_relu(d3)
        d3 = torch.cat([d3, skip3], dim=1)

        d2 = self.up2_transpose(d3)
        d2 = self.up2_relu(d2)
        d2 = torch.cat([d2, skip2], dim=1)

        d1 = self.up1_transpose(d2)
        out = self.out_activation(d1)

        return out

if __name__ == "__main__":
    autoencoder = UNet(in_channels=3, out_channels=3)

    image = torch.randn(1, 3, *IMG_SIZE)
    reconstructed = autoencoder(image)

    print("Input shape:       ", image.shape)
    print("Reconstructed shape:", reconstructed.shape)
