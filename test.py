import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from constant import MEAN, STD
from dataloader import LabeledImageDataset
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import PerceptualLoss
from constant import IMG_SIZE


def calculate_combined_loss(original, reconstructed, perceptual_loss_fn, mse_factor):

    mse_loss = F.mse_loss(reconstructed, original).item()


    perceptual_loss = perceptual_loss_fn(original, reconstructed).item()

    combined_loss = mse_factor * mse_loss + (1-mse_factor) * perceptual_loss
    return combined_loss

def evaluate_model_and_save_plot(model_path, dataset_path, labels_file, plot_save_path, thresholds=None, mse_factor=None):
    if thresholds is None:
        thresholds = [('example', 1.7022)]
    mean, std = MEAN, STD

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_set = LabeledImageDataset(root_dir=dataset_path, labels_file=labels_file, transform=transform)
    dataloader = DataLoader(test_set, batch_size=1, shuffle=False)


    test_labels = []

    model = torch.load(model_path)
    model.cuda()
    model.eval()

    perceptual_loss_fn = PerceptualLoss().cuda()

    loses = []
    loses_0 = []
    loses_1 = []
    for image, labels in tqdm(dataloader):
        test_labels.append(labels.item())

        with torch.no_grad():
            reconstructed = model(image.to('cuda'))

            loss = calculate_combined_loss(image.to('cuda'), reconstructed, perceptual_loss_fn, mse_factor)
            if labels.item() == 0:
                loses_0.append(loss)
            elif labels.item() == 1:
                loses_1.append(loss)
            loses.append(loss)

    for info, threshold in thresholds:
        predicted = []
        for loss in loses:
            if loss >= threshold:
                predicted.append(1)
            else:
                predicted.append(0)

        tps = sum(predicted[i] == 1 and test_labels[i] == 1 for i in range(len(predicted)))
        fps = sum(predicted[i] == 1 and test_labels[i] == 0 for i in range(len(predicted)))
        tns = sum(predicted[i] == 0 and test_labels[i] == 0 for i in range(len(predicted)))
        fns = sum(predicted[i] == 0 and test_labels[i] == 1 for i in range(len(predicted)))

        tpr = tps / (tps + fns) if (tps + fns) > 0 else 0
        tnr = tns / (tns + fps) if (tns + fps) > 0 else 0

        print('-'* 10)
        print(f'Info: {info}')
        print(f'Threshold: {threshold}:')
        print(f'True Positives: {tps}, False Positives: {fps}')
        print(f'True Negatives: {tns}, False Negatives: {fns}')
        print(f'True Positive Rate (TPR): {tpr:.4f}')
        print(f'True Negative Rate (TNR): {tnr:.4f}')
        print('-' * 10)

    plt.hist(loses_0, bins=100, alpha=0.5, label='List 1', edgecolor='black')
    plt.hist(loses_1 * 100, bins=100, alpha=0.5, label='List 2', edgecolor='black')

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Two Lists')
    plt.legend()
    if plot_save_path:
        plt.savefig(plot_save_path)
    plt.show()

if __name__ == '__main__':
    model_path = 'models/UNET_MSE_FACTOR_03.pth'
    dataset_path = 'dataset/test/imgs'
    labels_file = 'dataset/test/test_annotation.txt'
    mse_factor = 0.3

    threshold_range = [(f'Threshold {i}', i) for i in np.linspace(1, 2.0, 100)]

    evaluate_model_and_save_plot(
        model_path=model_path,
        dataset_path=dataset_path,
        labels_file=labels_file,
        plot_save_path=None,
        thresholds=threshold_range,
        mse_factor=mse_factor
    )
