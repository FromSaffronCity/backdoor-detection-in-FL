import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

def get_train_dataloader(poison_data=False):
    transform = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    dataset = CIFAR10(root="./", transform=transform, download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    return dataloader, len(dataset)

def get_test_dataloader(poison_data=False):
    transform = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    dataset = CIFAR10(root="./", train=False, transform=transform, download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=32)
    return dataloader, len(dataset)

def train(model, train_dataloader, num_epochs):
    if not model.training:
        model.train()

    criterion, optimizer = nn.CrossEntropyLoss(), Adam(params=model.parameters(), lr=0.001)

    for _ in range(num_epochs):
        for images, labels in train_dataloader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def test(model, test_dataloader):
    if model.training:
        model.eval()

    criterion = nn.CrossEntropyLoss()
    correct_predictions = total_images = loss = 0.0

    with torch.no_grad():
        for images, labels in test_dataloader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            loss = loss + criterion(outputs, labels).item()
            _, predictions = torch.max(input=outputs, dim=1)
            total_images = total_images + images.size(dim=0)
            correct_predictions = correct_predictions + torch.sum((predictions == labels), dtype=torch.uint8).item()

    accuracy = correct_predictions / total_images
    return loss, accuracy