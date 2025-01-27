import numpy as np
import torch

from constant import MEAN, STD, SEED
from constant import IMG_SIZE, MSE_FACTOR, NUM_EPOCH
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from dataloader import ImageFolderDataset
from tqdm import tqdm
from autoencoder import UNet
from utils import PerceptualLoss
from validate import visualize_results
import wandb

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


def autoencoder_loss_function(reconstructed, original, perceptual_loss_fn):
    mse_loss = F.mse_loss(reconstructed, original)
    perceptual_loss = perceptual_loss_fn(reconstructed, original)
    total_loss = MSE_FACTOR * mse_loss + (1 - MSE_FACTOR) * perceptual_loss
    return total_loss, mse_loss, perceptual_loss


def train_autoencoder(dataset_path, mean, std, image_size=IMG_SIZE,
                      batch_size=32, num_epochs=NUM_EPOCH, lr=1e-4):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset = ImageFolderDataset(dataset_path, transform=transform)

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    perceptual_loss_fn = PerceptualLoss().cuda()
    autoencoder = UNet().cuda()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)

    wandb.init(project="HW2-img-processing-course", name="UNET", config={
        "learning_rate": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "image_size": image_size
    })

    for epoch in range(num_epochs):
        autoencoder.train()
        epoch_loss = 0
        epoch_mse = 0
        epoch_perceptual_loss = 0

        for images, _ in tqdm(train_dataloader, desc=f'Training Autoencoder epoch {epoch + 1}/{num_epochs}...'):
            images = images.cuda()

            reconstructed = autoencoder(images)

            total_loss, mse_loss, perceptual_loss = autoencoder_loss_function(
                reconstructed, images, perceptual_loss_fn
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_mse += mse_loss.item()
            epoch_perceptual_loss += perceptual_loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        avg_mse = epoch_mse / len(train_dataloader)
        avg_perceptual_loss = epoch_perceptual_loss / len(train_dataloader)

        autoencoder.eval()
        val_loss = 0
        val_mse = 0
        val_perceptual_loss = 0
        with torch.no_grad():
            for images, _ in val_dataloader:
                images = images.cuda()
                reconstructed = autoencoder(images)
                total_loss, mse_loss, perceptual_loss = autoencoder_loss_function(
                    reconstructed, images, perceptual_loss_fn
                )
                val_loss += total_loss.item()
                val_mse += mse_loss.item()
                val_perceptual_loss += perceptual_loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_mse = val_mse / len(val_dataloader)
        avg_val_perceptual_loss = val_perceptual_loss / len(val_dataloader)

        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "mse_loss": avg_mse,
            "avg_perceptual_loss": avg_perceptual_loss,
            "val_loss": avg_val_loss,
            "val_mse_loss": avg_val_mse,
            "val_perceptual_loss": avg_val_perceptual_loss
        })

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Loss: {avg_loss:.4f}, \n"
              f"MSE: {avg_mse:.4f} \n"
              f"Perceptual Loss:{avg_perceptual_loss:.4f} \n")
        print(
            f"Validation Loss: {avg_val_loss:.4f}, \n"
            f"Validation MSE: {avg_val_mse:.4f},\n "
            f"Validation Perceptual Loss: {avg_val_perceptual_loss:.4f}\n")

    model_path = "models/Unet.pth"
    torch.save(autoencoder, model_path)
    print(f"Модель сохранена в {model_path}!")
    wandb.finish()
    return autoencoder, model_path


if __name__ == "__main__":
    dataset_path = "dataset/train"

    mean, std = MEAN, STD
    print(f"Средние значения: {mean}")
    print(f"Стандартные отклонения: {std}")

    autoencoder, model_path = train_autoencoder(dataset_path, mean, std)

    visualize_results(model_path, dataset_path, mean, std)