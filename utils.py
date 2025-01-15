import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from dataloader import ImageFolderDataset

IMG_SIZE = (32, 64)
MSE_FACTOR = 0.4

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

def gaussian_window(window_size, sigma):
    gauss = torch.tensor([(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    gauss = torch.exp(gauss)
    return gauss / gauss.sum()


def create_gaussian_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian_window(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.T
    _2D_window = _2D_window.float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=4, size_average=True):
    _, channel, height, width = img1.size()

    window = create_gaussian_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


def ms_ssim(img1, img2, window_size=11, size_average=True, weights=None):
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    levels = len(weights)
    msssim = []

    for _ in range(levels):
        ssim_value = ssim(img1, img2, window_size, size_average=False)
        msssim.append(ssim_value.mean(dim=(1, 2, 3)))

        img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
        img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)

    msssim = torch.stack(msssim, dim=0)
    msssim = (msssim ** torch.tensor(weights).to(img1.device)).prod(dim=0)

    return msssim.mean() if size_average else msssim

from torchvision.models import vgg16
import torch.nn.functional as F

from torchvision.models import alexnet

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        alex = alexnet(pretrained=True).features
        self.layers = nn.Sequential(*list(alex)[:-1]).eval()
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_feats = self.layers(x)
        y_feats = self.layers(y)
        return F.mse_loss(x_feats, y_feats, size_average=True)

if __name__ == "__main__":
    img1 = torch.rand(1, 3, 128, 128).to("cuda")
    img2 = torch.rand(1, 3, 128, 128).to("cuda")

    ssim_value = ssim(img1, img2)
    print("SSIM:", ssim_value.item())