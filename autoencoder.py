import torch
import torch.nn as nn

from utils import IMG_SIZE


class VAE(nn.Module):
    def __init__(self, img_channels=3, img_size=IMG_SIZE, latent_dim=128):
        super(VAE, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 128, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # (B, 128, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (B, 256, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # (B, 256, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # (B, 512, H/16, W/16)
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (B, 512, H/16, W/16)
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = None

        self.decoder_fc = None
        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),  # (B, 256, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # (B, 256, H/8, W/8)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),  # (B, 128, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # (B, 128, H/4, W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),  # (B, 64, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),  # (B, 32, H, W)
            nn.ReLU(),
            nn.Conv2d(32, img_channels, kernel_size=3, padding=1),  # (B, img_channels, H, W)
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        if self.fc_mu is None:
            in_features = x.shape[1]
            self.fc_mu = nn.Linear(in_features, self.latent_dim).to(x.device)

        mu = self.fc_mu(x)
        return mu

    def reparameterize(self, mu):
        std = torch.sqrt(torch.tensor(1.0, device=mu.device))
        eps = torch.randn_like(mu)
        return mu + eps * std

    def decode(self, z):
        if self.decoder_fc is None:
            self.decoder_fc = nn.Linear(self.latent_dim, 512 * 2 * 4).to(z.device)

        x = self.decoder_fc(z)
        x = x.view(x.size(0), 512, 2, 4)
        x = self.decoder_net(x)
        return x

    def forward(self, x):
        mu = self.encode(x)
        z = self.reparameterize(mu)
        reconstructed = self.decode(z)

        logvar = torch.full_like(mu, torch.log(torch.tensor(1.0, device=mu.device)))

        return reconstructed, mu, logvar


class AutoEncoderWithSkipConnections(nn.Module):
    def __init__(self, img_channels=3, img_size=IMG_SIZE):
        super(AutoEncoderWithSkipConnections, self).__init__()

        self.img_channels = img_channels
        self.img_size = img_size

        self.encoder1 = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, H/2, W/2)
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 128, H/4, W/4)
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (B, 256, H/8, W/8)
            nn.ReLU()
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # (B, 512, H/16, W/16)
            nn.ReLU()
        )

        # --- Декодер ---
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),  # (B, 256, H/8, W/8)
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),  # (B, 128, H/4, W/4)
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),  # (B, 64, H/2, W/2)
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, img_channels, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),  # (B, img_channels, H, W)
            nn.Sigmoid()  # Для нормализации значений в диапазон [0, 1]
        )

    def encode(self, x):

        skip1 = self.encoder1(x)  # (B, 64, H/2, W/2)
        skip2 = self.encoder2(skip1)  # (B, 128, H/4, W/4)
        skip3 = self.encoder3(skip2)  # (B, 256, H/8, W/8)
        encoded = self.encoder4(skip3)  # (B, 512, H/16, W/16)

        return encoded, (skip1, skip2, skip3)

    def decode(self, encoded, skips):

        skip1, skip2, skip3 = skips

        x = self.decoder4(encoded)  # (B, 256, H/8, W/8)
        x = x + skip3  # Сложение с skip-connection
        x = self.decoder3(x)  # (B, 128, H/4, W/4)
        x = x + skip2  # Сложение с skip-connection
        x = self.decoder2(x)  # (B, 64, H/2, W/2)
        x = x + skip1  # Сложение с skip-connection
        x = self.decoder1(x)  # (B, img_channels, H, W)

        return x

    def forward(self, x):
        encoded, skips = self.encode(x)
        reconstructed = self.decode(encoded, skips)
        return reconstructed

class AutoEncoderWithAttention(nn.Module):
    def __init__(self, img_channels=1, base_channels=64):

        super(AutoEncoderWithAttention, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, kernel_size=3, stride=2, padding=1),  # (B, base_channels, H/2, W/2)
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),  # (B, base_channels*2, H/4, W/4)
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),  # (B, base_channels*4, H/8, W/8)
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),  # (B, base_channels*8, H/16, W/16)
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, stride=2, padding=1),  # (B, base_channels*16, H/32, W/32)
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, base_channels*8, H/16, W/16)
            nn.ReLU(),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=1, padding=1),  # Промежуточное преобразование
            nn.ReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, base_channels*4, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1),  # Промежуточное преобразование
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, base_channels*2, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1),  # Промежуточное преобразование
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, base_channels, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),  # Промежуточное преобразование
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, img_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, img_channels, H, W)
            nn.Sigmoid()  # Нормализация
        )

    def encode(self, x):
        skip1 = self.encoder1(x)  # (B, base_channels, H/2, W/2)
        skip2 = self.encoder2(skip1)  # (B, base_channels*2, H/4, W/4)
        skip3 = self.encoder3(skip2)  # (B, base_channels*4, H/8, W/8)
        skip4 = self.encoder4(skip3)  # (B, base_channels*8, H/16, W/16)
        encoded = self.encoder5(skip4)  # (B, base_channels*16, H/32, W/32)
        return encoded, (skip1, skip2, skip3, skip4)

    def decode(self, encoded, skips):
        skip1, skip2, skip3, skip4 = skips

        x = self.decoder5(encoded)  # (B, base_channels*8, H/16, W/16)
        x = x + skip4  # Сложение с skip-connection
        x = self.decoder4(x)  # (B, base_channels*4, H/8, W/8)
        x = x + skip3  # Сложение с skip-connection
        x = self.decoder3(x)  # (B, base_channels*2, H/4, W/4)
        x = x + skip2  # Сложение с skip-connection
        x = self.decoder2(x)  # (B, base_channels, H/2, W/2)
        x = x + skip1  # Сложение с skip-connection
        x = self.decoder1(x)  # (B, img_channels, H, W)

        return x

    def forward(self, x):
        encoded, skips = self.encode(x)
        reconstructed = self.decode(encoded, skips)
        return reconstructed


if __name__ == "__main__":
    img_size = IMG_SIZE
    autoencoder = AutoEncoderWithAttention(img_channels=3)

    image = torch.randn(1, 3, *img_size)
    reconstructed = autoencoder(image)

    print("Input shape:       ", image.shape)
    print("Reconstructed shape:", reconstructed.shape)
