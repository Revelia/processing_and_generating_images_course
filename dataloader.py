import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Создаем собственный класс Dataset
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path


class LabeledImageDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Читаем разметку из файла
        with open(labels_file, 'r') as f:
            self.labels = [
                line.strip().split() for line in f.readlines()
            ]


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image_name, label = self.labels[idx]
        image_path = os.path.join(self.root_dir, image_name)
        label = int(label)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Пример использования
if __name__ == "__main__":
    root_dir = "dataset/train"
    mean = [0.42343804240226746, 0.5342181324958801, 0.4620889723300934]
    std = [0.04519784078001976, 0.05054565891623497, 0.046623989939689636]
    transform = transforms.Compose([
        transforms.Resize((49,25)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset = ImageFolderDataset(root_dir, transform=transform)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for images, paths in dataloader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Paths: {paths}")

        break