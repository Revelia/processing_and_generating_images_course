import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloader import ImageFolderDataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import PerceptualLoss
from constant import IMG_SIZE, MSE_FACTOR



def compute_combined_loss_for_dataset(autoencoder, dataloader, device, perceptual_loss_fn):
    autoencoder.eval()
    all_losses = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Вычисление комбинированного лосса для датасета..."):
            images = images.to(device)
            reconstructed = autoencoder(images)

            mse_loss = F.mse_loss(reconstructed, images).item()
            perceptual_loss = perceptual_loss_fn(images, reconstructed).item()

            combined_loss = MSE_FACTOR * mse_loss + (1-MSE_FACTOR) * perceptual_loss
            all_losses.append(combined_loss)

    return np.array(all_losses)


def find_optimal_threshold(loss_values, labels, threshold_range=np.linspace(0.01, 1, 100)):
    best_f1 = 0
    best_threshold = 0
    metrics = []

    for threshold in threshold_range:
        predictions = (loss_values > threshold).astype(int)

        f1 = f1_score(labels, predictions)

        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics.append({
            "threshold": threshold,
            "f1": f1,
            "tpr": tpr,
            "tnr": tnr
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1, metrics


def plot_metrics(metrics, save_path="metrics_plot.png"):
    thresholds = [m["threshold"] for m in metrics]
    f1_scores = [m["f1"] for m in metrics]
    tpr_values = [m["tpr"] for m in metrics]
    tnr_values = [m["tnr"] for m in metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label="F1-Score", marker="o")
    plt.plot(thresholds, tpr_values, label="TPR (True Positive Rate)", marker="o")
    plt.plot(thresholds, tnr_values, label="TNR (True Negative Rate)", marker="o")

    plt.title("Метрики в зависимости от порога")
    plt.xlabel("Порог")
    plt.ylabel("Значение метрики")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.show()
    print(f"График сохранён в {save_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = IMG_SIZE
    mean = [0.42339497804641724, 0.5341755151748657, 0.46204590797424316]
    std = [0.04559207707643509, 0.05092834308743477, 0.047043971717357635]

    autoencoder = torch.load("models/autoencoder_attentive.pth").to(device)
    autoencoder.eval()

    perceptual_loss_fn = PerceptualLoss().to(device)

    # Трансформации
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = ImageFolderDataset("dataset/train", transform=transform)
    proliv_dataset = ImageFolderDataset("dataset/proliv", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    proliv_loader = DataLoader(proliv_dataset, batch_size=1, shuffle=False)

    loss_train = compute_combined_loss_for_dataset(autoencoder, train_loader, device, perceptual_loss_fn)
    loss_proliv = compute_combined_loss_for_dataset(autoencoder, proliv_loader, device, perceptual_loss_fn)

    loss_values = np.concatenate([loss_train, loss_proliv])
    labels = np.concatenate([np.zeros(len(loss_train)), np.ones(len(loss_proliv))])

    threshold_range = np.linspace(1, 10, 1000)
    optimal_threshold, best_f1, metrics = find_optimal_threshold(loss_values, labels, threshold_range)

    print(f"Оптимальный порог: {optimal_threshold:.4f}, F1: {best_f1:.4f}")
    for metric in metrics:
        print(f"Порог: {metric['threshold']:.4f}, F1: {metric['f1']:.4f}, TPR: {metric['tpr']:.4f}, TNR: {metric['tnr']:.4f}")

    # Построение графиков
    plot_metrics(metrics, save_path="threshold_metrics_combined_loss.png")