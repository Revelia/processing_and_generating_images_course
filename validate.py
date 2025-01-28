import os

import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from constant import IMG_SIZE, MEAN, STD
from dataloader import ImageFolderDataset
from utils import calculate_mean_std, PerceptualLoss, denormalize
import torch.nn.functional as F


def visualize_results(model_path, dataset_path, mean, std, image_size=IMG_SIZE, save_path="results.png"):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)

    ])
    dataset = ImageFolderDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=False)


    autoencoder = torch.load(model_path)
    autoencoder.eval()
    print(f"Модель успешно загружена из {model_path}")

    images, _ = next(iter(dataloader))
    images = images.cuda()

    with torch.no_grad():
        reconstructed = autoencoder(images)

    reconstructed = denormalize(reconstructed, mean, std).cpu().numpy().transpose(0, 2, 3, 1)
    images = denormalize(images, mean, std).cpu().numpy().transpose(0, 2, 3, 1)

    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(2, 6, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.subplot(2, 6, i + 7)
        plt.imshow(reconstructed[i])
        plt.axis("off")
    plt.savefig(save_path)
    plt.show()


def calculate_loss_distribution(model_path,
                                dataset_path,
                                mean,
                                std,
                                image_size=IMG_SIZE,
                                save_path="loss_distribution.png",
                                mse_factor= None):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    dataset = ImageFolderDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    autoencoder = torch.load(model_path)
    autoencoder.eval()

    perceptual_loss_fn = PerceptualLoss().cuda()
    combined_losses = []

    for images, _ in tqdm(dataloader, desc="Вычисление комбинированного лосса"):
        images = images.cuda()
        with torch.no_grad():
            reconstructed = autoencoder(images)


            mse_loss = F.mse_loss(reconstructed, images).item()

            perceptual_loss = perceptual_loss_fn(images, reconstructed).item()

            combined_loss = mse_factor * mse_loss + (1-mse_factor) * perceptual_loss
            combined_losses.append(combined_loss)

    plt.figure(figsize=(10, 6))
    plt.hist(combined_losses, bins=30, color="purple", alpha=0.7, edgecolor="black")
    plt.title("Распределение комбинированного лосса (MSE + Perceptual Loss)")
    plt.xlabel("Combined Loss")
    plt.ylabel("Частота")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig(save_path)
    plt.show()
    print(f"График распределения комбинированного лосса сохранён в {save_path}")

def validate(model_path='models/Unet.pth', experiment_name=None, mse_factor=0.5):


    dataset_path = 'dataset/val'
    dataset_proliv = 'dataset/proliv'

    mean, std = MEAN, STD
    model_path = model_path

    base_save_path = experiment_name if experiment_name else ""

    if experiment_name and not os.path.exists(base_save_path):
        os.makedirs(base_save_path)

    results_train_path = os.path.join(base_save_path, "results_train.png")
    results_proliv_path = os.path.join(base_save_path, "results_proliv.png")
    loss_distribution_train_path = os.path.join(base_save_path, "loss_distribution_train.png")
    loss_distribution_proliv_path = os.path.join(base_save_path, "loss_distribution_proliv.png")

    visualize_results(model_path, dataset_path, mean, std, save_path=results_train_path)
    visualize_results(model_path, dataset_proliv, mean, std, save_path=results_proliv_path)

    calculate_loss_distribution(
        model_path=model_path,
        dataset_path=dataset_path,
        mean=mean,
        std=std,
        save_path=loss_distribution_train_path,
        mse_factor=mse_factor,
    )

    calculate_loss_distribution(
        model_path=model_path,
        dataset_path=dataset_proliv,
        mean=mean,
        std=std,
        save_path=loss_distribution_proliv_path,
        mse_factor=mse_factor,
    )

if __name__ == "__main__":
    validate()
