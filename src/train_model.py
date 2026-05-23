from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "photo_classifier.pth"

BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(DATASET_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATASET_DIR / "val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, train_dataset.classes

def create_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Сначала замораживаем все слои
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем последний блок ResNet, чтобы модель лучше адаптировалась к нашему датасету
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Меняем последний слой под наши 2 класса
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model.to(device)


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_names = create_dataloaders()

    print("Классы:", class_names)
    print("Устройство:", device)

    model = create_model(num_classes=len(class_names))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    best_val_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nЭпоха {epoch + 1}/{NUM_EPOCHS}")

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = val_correct / val_total

        print(f"Train loss: {train_loss / len(train_loader):.4f}")
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Val loss: {val_loss / len(val_loader):.4f}")
        print(f"Val accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names
            }, MODEL_PATH)

            print(f"Модель сохранена: {MODEL_PATH}")

    print("\nОбучение завершено")
    print(f"Лучшая точность на val: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    train()