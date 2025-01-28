import random

import numpy as np
import torch

from constant import MEAN, STD, SEED, BATCH_SIZE, LR
from constant import IMG_SIZE, NUM_EPOCH
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import ImageFolderDataset
from tqdm import tqdm
from autoencoder import UNet
from utils import PerceptualLoss
import wandb
import matplotlib.pyplot as plt
import cv2

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)



def apply_artifacts(images):
    images_clone = images.clone()
    _, _, height, width = images_clone.size()
    circle_radius = min(height, width) // 4

    for i in range(images_clone.size(0)):
        if random.random() < 0.3:

            offset_x = random.randint(-circle_radius, circle_radius)
            offset_y = random.randint(-circle_radius, circle_radius)
            center_x = width // 2 + offset_x
            center_y = height // 2 + offset_y


            mask = np.zeros((height, width), dtype=np.uint8)

            cv2.circle(mask, (center_x, center_y), circle_radius, 1, thickness=-1)


            image_np = images_clone[i].cpu().numpy().transpose(1, 2, 0)

            image_np = np.ascontiguousarray(image_np)


            image_np[mask == 1] = np.random.rand() * image_np[mask == 1] + 255 * np.random.rand()


            images_clone[i] = torch.from_numpy(image_np.transpose(2, 0, 1))

    return images_clone

def autoencoder_loss_function(reconstructed, original, perceptual_loss_fn, mse_factor):
    mse_loss = F.mse_loss(reconstructed, original)
    perceptual_loss = perceptual_loss_fn(reconstructed, original)
    total_loss = mse_factor * mse_loss + (1 - mse_factor) * perceptual_loss
    return total_loss, mse_loss, perceptual_loss


def train_autoencoder(dataset_path, val_dataset_path, mean, std, model_path = "models/Unet.pth", experiment_name="UNET", mse_factor=None):

    image_size = IMG_SIZE
    batch_size = BATCH_SIZE
    num_epochs = NUM_EPOCH
    lr = LR
    mse_factor = mse_factor

    print(f"Train model with image size: {image_size}, batch size: {batch_size}, num epochs: {num_epochs}, learning rate: {lr}, mse_factor: {mse_factor}")

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = ImageFolderDataset(dataset_path, transform=transform)
    val_dataset = ImageFolderDataset(val_dataset_path, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    perceptual_loss_fn = PerceptualLoss().cuda()
    autoencoder = UNet().cuda()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)

    wandb.init(project="HW2-img-processing-course", name=experiment_name, config={
        "learning_rate": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "image_size": image_size,
        "mse_factor": mse_factor,
    })

    for epoch in range(num_epochs):
        autoencoder.train()
        epoch_loss = 0
        epoch_mse = 0
        epoch_perceptual_loss = 0

        for images, _ in tqdm(train_dataloader, desc=f'Training Autoencoder epoch {epoch + 1}/{num_epochs}...'):
            images_with_artifacts = apply_artifacts(images)
            images_with_artifacts = images_with_artifacts.cuda()
            images = images.cuda()

            reconstructed = autoencoder(images_with_artifacts)

            total_loss, mse_loss, perceptual_loss = autoencoder_loss_function(
                reconstructed, images, perceptual_loss_fn, mse_factor
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
                    reconstructed, images, perceptual_loss_fn, mse_factor
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

        print(f"Epoch [{epoch + 1}/{num_epochs}], \n"
              f"Loss: {avg_loss:.4f}, \n"
              f"MSE: {avg_mse:.4f} \n"
              f"Perceptual Loss: {avg_perceptual_loss:.4f} \n")
        print(
            f"Validation Loss: {avg_val_loss:.4f},\n"
            f"Validation MSE: {avg_val_mse:.4f},\n"
            f"Validation Perceptual Loss: {avg_val_perceptual_loss:.4f}\n")

    torch.save(autoencoder, model_path)
    print(f"Модель сохранена в {model_path}!")
    wandb.finish()
    return autoencoder, model_path


if __name__ == "__main__":
    dataset_path = "dataset/train"
    val_dataset_path = "dataset/val"

    mean, std = MEAN, STD
    print(f"Средние значения: {mean}")
    print(f"Стандартные отклонения: {std}")


    # Visualize the effect of artifacts
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    dataset = ImageFolderDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=False)

    images, _ = next(iter(dataloader))
    images_with_artifacts = apply_artifacts(images)

    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images_with_artifacts = images_with_artifacts.permute(0, 2, 3, 1).cpu().numpy()

    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(2, 6, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.subplot(2, 6, i + 7)
        plt.imshow(images_with_artifacts[i])
        plt.axis("off")
    plt.show()

    # dataset = ImageFolderDataset("dataset/proliv", transform=transform)
    # dataloader = DataLoader(dataset, batch_size=6, shuffle=False)
    #
    # images, _ = next(iter(dataloader))
    # images_with_artifacts = apply_artifacts(images)
    #
    # images = images.permute(0, 2, 3, 1).cpu().numpy()
    # images_with_artifacts = images_with_artifacts.permute(0, 2, 3, 1).cpu().numpy()
    #
    # plt.figure(figsize=(12, 6))
    # for i in range(6):
    #     plt.subplot(2, 6, i + 1)
    #     plt.imshow(images[i])
    #     plt.axis("off")
    #     plt.subplot(2, 6, i + 7)
    #     plt.imshow(images_with_artifacts[i])
    #     plt.axis("off")
    # plt.show()