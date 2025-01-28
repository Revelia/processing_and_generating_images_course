import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from constant import IMG_SIZE, SEED
from dataloader import ImageFolderDataset
from torchvision.models import alexnet, vgg16
import torch.nn.functional as F

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


def calculate_mean_std(dataset_path, image_size=IMG_SIZE):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    dataset = ImageFolderDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    channel_sum = torch.zeros(3)
    channel_sum_squared = torch.zeros(3)
    num_pixels = 0

    for images, _ in tqdm(dataloader, desc='Computing mean and std'):
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_squared += (images ** 2).sum(dim=[0, 2, 3])
        num_pixels += images.size(0) * images.size(2) * images.size(3)

    mean = channel_sum / num_pixels
    std = ((channel_sum_squared / num_pixels) - (mean ** 2)).sqrt()

    return mean.tolist(), std.tolist()

def denormalize(image, mean, std):
    mean = torch.tensor(mean, device=image.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=image.device).view(1, -1, 1, 1)
    image = image * std + mean
    return image



class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        alex = vgg16(pretrained=True).features
        self.layers = nn.Sequential(*list(alex)[:-1]).eval()
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_feats = self.layers(x)
        y_feats = self.layers(y)
        return F.mse_loss(x_feats, y_feats, size_average=True)
