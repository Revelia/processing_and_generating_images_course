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
from constant import IMG_SIZE



def compute_combined_loss_for_dataset(autoencoder, dataloader, device, perceptual_loss_fn, mse_factor):
    autoencoder.eval()
    all_losses = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Вычисление комбинированного лосса для датасета..."):
            images = images.to(device)
            reconstructed = autoencoder(images)

            mse_loss = F.mse_loss(reconstructed, images).item()
            perceptual_loss = perceptual_loss_fn(images, reconstructed).item()

            combined_loss = mse_factor * mse_loss + (1-mse_factor) * perceptual_loss
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


def find_and_save_optimal_threshold(model_path,
                                    val_dataset_path,
                                    proliv_dataset_path,
                                    mean, std,
                                    device,
                                    save_path, metrics_file_path,
                                    mse_factor):
    autoencoder = torch.load(model_path)

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = ImageFolderDataset(val_dataset_path, transform=transform)
    proliv_dataset = ImageFolderDataset(proliv_dataset_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    proliv_loader = DataLoader(proliv_dataset, batch_size=1, shuffle=False)

    perceptual_loss_fn = PerceptualLoss().to(device)
    loss_train = compute_combined_loss_for_dataset(autoencoder, train_loader, device, perceptual_loss_fn, mse_factor)
    loss_proliv = compute_combined_loss_for_dataset(autoencoder, proliv_loader, device, perceptual_loss_fn, mse_factor)

    loss_values = np.concatenate([loss_train, loss_proliv])
    labels = np.concatenate([np.zeros(len(loss_train)), np.ones(len(loss_proliv))])
    threshold_range = np.linspace(0, 10, 10000)
    optimal_threshold, best_f1, metrics = find_optimal_threshold(loss_values, labels, threshold_range)

    print(f"Оптимальный порог: {optimal_threshold:.4f}, F1: {best_f1:.4f}")

    plot_metrics(metrics, save_path=save_path)

    with open(metrics_file_path, 'w') as f:
        f.write("Threshold\tF1\tTPR\tTNR\n")
        for metric in metrics:
            f.write(f"{metric['threshold']:.4f}\t{metric['f1']:.4f}\t{metric['tpr']:.4f}\t{metric['tnr']:.4f}\n")

    best_f1_threshold = max(metrics, key=lambda x: x['f1'])['threshold']
    tpr_95_threshold = next((m['threshold'] for m in metrics if m['tpr'] <= 0.95), None)
    tnr_95_threshold = next((m['threshold'] for m in metrics if m['tnr'] >= 0.95), None)

    print(f"Threshold maximizing F1: {best_f1_threshold:.4f}")
    print(f"Threshold with TPR >= 0.95: {tpr_95_threshold:.4f}")
    print(f"Threshold with TNR >= 0.95: {tnr_95_threshold:.4f}")

    # Save the best thresholds to the text file
    with open(metrics_file_path, 'a') as f:
        f.write("Threshold\tF1\tTPR\tTNR\n")
        for metric in metrics:
            f.write(f"{metric['threshold']:.4f}\t{metric['f1']:.4f}\t{metric['tpr']:.4f}\t{metric['tnr']:.4f}\n")
        f.write("\nBest Thresholds:\n")
        f.write(f"Max F1 Threshold: {best_f1_threshold:.4f}\n")
        f.write(f"TPR >= 0.95 Threshold: {tpr_95_threshold:.4f}\n")
        f.write(f"TNR >= 0.95 Threshold: {tnr_95_threshold:.4f}\n")

    return [('Best F1:', best_f1_threshold),
            ('TPR 95:', tpr_95_threshold),
            ('TNR 95:', tnr_95_threshold),
            ('TNR TPR AVG:', (tpr_95_threshold + tnr_95_threshold)/2.0)]
