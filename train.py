import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import wandb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from constants import SEED
from model import ResNet18, ViT



torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

WANDB_TOKEN = os.environ['WANDB_TOKEN']

wandb.login(key=WANDB_TOKEN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

model = ViT().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def train(epochs):

    wandb.init(project="HW1-fashion-mnist", name="ViT-bigger", config={
        "epochs": epochs,
        "batch_size": 64,
        "learning_rate": 1e-5,
        "architecture": "ResNet18"
        }
    )
    epoch_list = []
    precision_list = [[] for _ in range(10)]
    recall_list = [[] for _ in range(10)]
    f1_list = [[] for _ in range(10)]
    best_val_loss = float('inf')
    best_model_weights = None
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        correct = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                y_true.extend(target.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        accuracy, precision, recall, f1 = compute_metrics(y_true, y_pred)

        print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} '
              f'({100. * correct / len(val_loader.dataset):.2f}%)')

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()

        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

        epoch_list.append(epoch)
        for i in range(10):
            precision_list[i].append(class_precision[i])
            recall_list[i].append(class_recall[i])
            f1_list[i].append(class_f1[i])

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        for i in range(10):
            ax1.plot(epoch_list, precision_list[i], label=f'Class {i}')
            ax2.plot(epoch_list, recall_list[i], label=f'Class {i}')
            ax3.plot(epoch_list, f1_list[i], label=f'Class {i}')

        ax1.set_title('Precision per class')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Precision')
        ax1.legend()

        ax2.set_title('Recall per class')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Recall')
        ax2.legend()

        ax3.set_title('F1-score per class')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1-score')
        ax3.legend()

        plt.tight_layout()

        wandb.log({
            "Metrics per class": wandb.Image(fig)
        })

        plt.close(fig)

    model.load_state_dict(best_model_weights)


def evaluate():
    model.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy, precision, recall, f1 = compute_metrics(y_true, y_pred)

    print(f'\nTest set: Average loss: {test_loss:.4f} \n'
          f' Accuracy: {accuracy}\n'
          f' Precision: {precision}\n'
          f' Recall: {recall}\n'
          f' F1_score: {f1}\n')


if __name__ == "__main__":
    train(epochs=50)
    evaluate()
    wandb.finish()