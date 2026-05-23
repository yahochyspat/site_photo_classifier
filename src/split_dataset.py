import os
import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SORTED_DIR = PROJECT_ROOT / "data" / "sorted"
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"

CLASSES = ["photo", "not_photo"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

random.seed(42)


def clear_dataset_dir():
    for split in ["train", "val", "test"]:
        for class_name in CLASSES:
            folder = DATASET_DIR / split / class_name
            folder.mkdir(parents=True, exist_ok=True)

            for file in folder.iterdir():
                if file.is_file():
                    file.unlink()


def split_class(class_name):
    source_dir = SORTED_DIR / class_name

    files = [
        file for file in source_dir.iterdir()
        if file.is_file() and file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ]

    random.shuffle(files)

    total = len(files)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split, split_files in split_map.items():
        target_dir = DATASET_DIR / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for file in split_files:
            shutil.copy(file, target_dir / file.name)

    print(f"{class_name}: всего {total}")
    print(f"  train: {len(train_files)}")
    print(f"  val:   {len(val_files)}")
    print(f"  test:  {len(test_files)}")


def main():
    clear_dataset_dir()

    for class_name in CLASSES:
        split_class(class_name)

    print("Датасет успешно разделён на train / val / test")


if __name__ == "__main__":
    main()