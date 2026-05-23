from pathlib import Path
import shutil

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from parser_united import download_images_combined


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "photo_classifier.pth"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_PHOTO_DIR = RESULTS_DIR / "photo"
RESULTS_NOT_PHOTO_DIR = RESULTS_DIR / "not_photo"
REPORT_PATH = RESULTS_DIR / "filter_report.txt"

PHOTO_THRESHOLD = 0.80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names


def predict_image(model, class_names, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()

    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index].item()

    return predicted_class, confidence


def clear_folder(folder):
    folder.mkdir(parents=True, exist_ok=True)

    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def main():
    site_url = input("Введите URL сайта: ")

    clear_folder(RAW_DIR)
    clear_folder(RESULTS_DIR)

    RESULTS_PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_NOT_PHOTO_DIR.mkdir(parents=True, exist_ok=True)

    print("Скачиваем изображения...")

    try:
        downloaded_files = download_images_combined(site_url, output_dir=RAW_DIR)
    except Exception as error:
        print(f"Не удалось обработать сайт: {error}")
        return

    print("\nЗагружаем модель...")
    model, class_names = load_model()

    print("Классы модели:", class_names)

    photo_count = 0
    not_photo_count = 0
    report_lines = []

    report_lines.append(f"URL сайта: {site_url}")
    report_lines.append(f"Всего скачано изображений: {len(downloaded_files)}")
    report_lines.append("")
    report_lines.append("Результаты классификации:")
    report_lines.append("filename; predicted_class; confidence; saved_to")

    print("\nФильтруем изображения...")

    for image_path in downloaded_files:
        image_path = Path(image_path)

        try:
            predicted_class, confidence = predict_image(model, class_names, image_path)

            # Если модель сказала photo, но уверенность ниже порога,
            # считаем изображение not_photo.
            if predicted_class == "photo" and confidence >= PHOTO_THRESHOLD:
                target_dir = RESULTS_PHOTO_DIR
                final_class = "photo"
                photo_count += 1
            else:
                target_dir = RESULTS_NOT_PHOTO_DIR
                final_class = "not_photo"
                not_photo_count += 1

            shutil.copy(image_path, target_dir / image_path.name)

            print(
                f"{image_path.name}: predicted={predicted_class}, "
                f"confidence={confidence:.3f}, saved_to={final_class}"
            )

            report_lines.append(
                f"{image_path.name}; {predicted_class}; {confidence:.3f}; {final_class}"
            )

        except Exception as error:
            print(f"Ошибка с файлом {image_path.name}: {error}")
            report_lines.append(f"{image_path.name}; error; {error}; not_saved")

    report_lines.append("")
    report_lines.append(f"Сохранено в photo: {photo_count}")
    report_lines.append(f"Сохранено в not_photo: {not_photo_count}")
    report_lines.append(f"Папка с результатами: {RESULTS_DIR}")

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines))

    print("\nГотово.")
    print(f"Всего скачано: {len(downloaded_files)}")
    print(f"Сохранено в photo: {photo_count}")
    print(f"Сохранено в not_photo: {not_photo_count}")
    print(f"Результаты находятся в папке: {RESULTS_DIR}")
    print(f"Отчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    main()