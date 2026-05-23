from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "photo_classifier.pth"

BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(num_classes):
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint["class_names"]


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = datasets.ImageFolder(DATASET_DIR / "test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model, class_names = load_model(num_classes=len(test_dataset.classes))

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    print("Классы:", class_names)
    print("\nConfusion matrix:")
    print(confusion_matrix(all_labels, all_predictions))

    print("\nClassification report:")
    print(classification_report(
        all_labels,
        all_predictions,
        target_names=class_names
    ))


if __name__ == "__main__":
    main()