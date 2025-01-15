import os
import torch
import torch.optim as optim
import torch.nn as nn
from utils import IMG_SIZE, MSE_FACTOR
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import ImageFolderDataset
from tqdm import tqdm
from autoencoder import AutoEncoderWithSkipConnections, AutoEncoderWithAttention  # Импортируем автоэнкодер
from utils import calculate_mean_std, ms_ssim, ssim, PerceptualLoss
from validate import visualize_results
import wandb  # Импортируем WandB
import pytorch_ssim

NUM_EPOCH = 20

def autoencoder_loss_function(reconstructed, original, perceptual_loss_fn):
    mse_loss = F.mse_loss(reconstructed, original)  # MSE между реконструированным и исходным
    perceptual_loss = perceptual_loss_fn(reconstructed, original)  # Перцептуальная потеря
    total_loss = MSE_FACTOR * mse_loss + (1- MSE_FACTOR) * perceptual_loss  # Взвешенное комбинирование потерь
    return total_loss, mse_loss, perceptual_loss


def train_autoencoder(dataset_path, mean, std, image_size=IMG_SIZE,
                      batch_size=32, num_epochs=NUM_EPOCH, lr=1e-4):
    transform = transforms.Compose([
        # transforms.Resize(image_size),  # Изменение размера изображения
        transforms.RandomCrop(IMG_SIZE, pad_if_needed=True, padding_mode ='symmetric'),
        transforms.RandomHorizontalFlip(p=0.5),  # Случайное горизонтальное отражение
        transforms.RandomVerticalFlip(p=0.5),  # Случайное вертикальное отражение
        transforms.ToTensor(),  # Преобразуем в тензор

        transforms.Normalize(mean, std)  # Нормализация
    ])

    dataset = ImageFolderDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    perceptual_loss_fn = PerceptualLoss().cuda()
    autoencoder = AutoEncoderWithAttention(img_channels=3).cuda()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)

    wandb.init(project="autoencoder-training", name="Autoencoder", config={
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

        for images, _ in tqdm(dataloader, desc=f'Training Autoencoder epoch {epoch + 1}/{num_epochs}...'):
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

        avg_loss = epoch_loss / len(dataloader)
        avg_mse = epoch_mse / len(dataloader)
        avg_perceptual_loss = epoch_perceptual_loss / len(dataloader)

        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "mse_loss": avg_mse,
            "avg_perceptual_loss": avg_perceptual_loss
        })

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}")

    model_path = "models/autoencoder_attentive.pth"
    torch.save(autoencoder, model_path)
    print(f"Модель сохранена в {model_path}!")
    wandb.finish()
    return autoencoder, model_path


if __name__ == "__main__":
    dataset_path = "dataset/train"

    mean, std = calculate_mean_std(dataset_path, image_size=IMG_SIZE)
    print(f"Средние значения: {mean}")
    print(f"Стандартные отклонения: {std}")

    autoencoder, model_path = train_autoencoder(dataset_path, mean, std)

    visualize_results(model_path, dataset_path, mean, std)