import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from dataloader import LabeledImageDataset
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import calculate_mean_std, ssim, PerceptualLoss, IMG_SIZE, MSE_FACTOR


def calculate_combined_loss(original, reconstructed, perceptual_loss_fn, mean, std):

    mse_loss = F.mse_loss(reconstructed, original).item()


    perceptual_loss = perceptual_loss_fn(original, reconstructed).item()

    combined_loss = MSE_FACTOR * mse_loss + (1-MSE_FACTOR) * perceptual_loss
    return combined_loss

if __name__ == "__main__":
    mean, std = calculate_mean_std('dataset/train', image_size=(64, 32))

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_set = LabeledImageDataset(root_dir='dataset/test/imgs',
                                   labels_file='dataset/test/test_annotation.txt',
                                   transform=transform)
    tsh = 1.8658

    dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    predicted = []
    test_labels = []

    model = torch.load('models/autoencoder_attentive.pth')
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

            loss = calculate_combined_loss(image.to('cuda'), reconstructed, perceptual_loss_fn, mean, std)
            if labels.item() == 0:
                loses_0.append(loss)
            elif labels.item() == 1:
                loses_1.append(loss)
            loses.append(loss)

            if loss >= tsh:
                predicted.append(1)
            else:
                predicted.append(0)

    tps = sum(predicted[i] == 1 and test_labels[i] == 1 for i in range(len(predicted)))
    fps = sum(predicted[i] == 1 and test_labels[i] == 0 for i in range(len(predicted)))
    tns = sum(predicted[i] == 0 and test_labels[i] == 0 for i in range(len(predicted)))
    fns = sum(predicted[i] == 0 and test_labels[i] == 1 for i in range(len(predicted)))

    tpr = tps / (tps + fns) if (tps + fns) > 0 else 0
    tnr = tns / (tns + fps) if (tns + fps) > 0 else 0

    print(f'True Positives: {tps}, False Positives: {fps}')
    print(f'True Negatives: {tns}, False Negatives: {fns}')
    print(f'True Positive Rate (TPR): {tpr:.4f}')
    print(f'True Negative Rate (TNR): {tnr:.4f}')

    plt.hist(loses_0, bins=100, alpha=0.5, label='List 1', edgecolor='black')
    plt.hist(loses_1, bins=100, alpha=0.5, label='List 2', edgecolor='black')

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Two Lists')
    plt.legend()

    plt.show()