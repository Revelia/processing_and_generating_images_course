import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import IMG_SIZE, MSE_FACTOR
from autoencoder import AutoEncoderWithSkipConnections
from dataloader import ImageFolderDataset
from utils import denormalize, calculate_mean_std, ssim, PerceptualLoss
import torch.nn.functional as F


def visualize_results(model_path, dataset_path, mean, std, image_size=IMG_SIZE):
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
    plt.show()


def calculate_loss_distribution(model_path,
                                dataset_path,
                                mean,
                                std,
                                image_size=IMG_SIZE,
                                save_path="loss_distribution.png"):
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

            combined_loss = MSE_FACTOR * mse_loss + (1-MSE_FACTOR) * perceptual_loss
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


if __name__ == "__main__":
    dataset_path = 'dataset/train'
    mean, std = calculate_mean_std(dataset_path, image_size=IMG_SIZE)
    model_path = 'models/autoencoder.pth'
    dataset_proliv = 'dataset/proliv'

    visualize_results(model_path, dataset_path, mean, std)
    visualize_results(model_path, dataset_proliv, mean, std)

    calculate_loss_distribution(
        model_path=model_path,
        dataset_path=dataset_path,
        mean=mean,
        std=std,
        save_path="loss_distribution_train.png"
    )

    # Считаем распределение потерь для набора 'proliv'
    calculate_loss_distribution(
        model_path=model_path,
        dataset_path=dataset_proliv,
        mean=mean,
        std=std,
        save_path="loss_distribution_proliv.png"
    )
