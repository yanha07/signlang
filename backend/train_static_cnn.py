import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 64


def load_static_dataset(data_root: Path):
    classes = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    images, labels = [], []

    for label_idx, action in enumerate(classes):
        action_dir = data_root / action
        for image_path in action_dir.iterdir():
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img.astype(np.float32) / 255.0)
            labels.append(label_idx)

    if not images:
        raise RuntimeError("No valid static images found.")

    x = np.asarray(images, dtype=np.float32)
    y = to_categorical(labels, num_classes=len(classes))
    return x, y, classes


def build_model(num_classes: int):
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train CNN model for static sign recognition.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/static"), help="Folder containing class/image files")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    x, y, classes = load_static_dataset(args.data_dir)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42, stratify=np.argmax(y, axis=1))

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(num_classes=len(classes))
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ModelCheckpoint(args.model_dir / "static_cnn.keras", monitor="val_loss", save_best_only=True, verbose=1),
        ],
    )

    np.save(args.model_dir / "static_labels.npy", np.array(classes, dtype=object))
    print("Saved static model and labels.")


if __name__ == "__main__":
    main()
