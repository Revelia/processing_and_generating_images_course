import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ResNet18
import wandb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt


wandb.login(key="")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
wandb.init(project="fashion-mnist-resnet", name="resnet18")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)


model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def train(epochs):

    wandb.init(project="fashion-mnist-resnet", name="resnet18", config={
        "epochs": epochs,
        "batch_size": 64,
        "learning_rate": 0.001,
        "architecture": "ResNet18"
        }
    )
    epoch_list = []
    precision_list = [[] for _ in range(10)]
    recall_list = [[] for _ in range(10)]
    f1_list = [[] for _ in range(10)]
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                wandb.log({"train_loss": loss.item()})

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

        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
              f'({100. * correct / len(test_loader.dataset):.2f}%)')

        wandb.log({
            "test_loss": test_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

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


if __name__ == "__main__":

    train(epochs=10)
    wandb.finish()