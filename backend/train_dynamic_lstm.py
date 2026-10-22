import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

SEQUENCE_LENGTH = 30
FRAME_SIZE = 1662


def load_dynamic_dataset(data_root: Path):
    classes = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    sequences, labels = [], []

    for label_idx, action in enumerate(classes):
        action_dir = data_root / action
        for seq_dir in sorted([d for d in action_dir.iterdir() if d.is_dir()]):
            frames = []
            valid = True
            for frame_id in range(SEQUENCE_LENGTH):
                frame_file = seq_dir / f"{frame_id}.npy"
                if not frame_file.exists():
                    valid = False
                    break
                frame = np.load(frame_file)
                if frame.shape[0] != FRAME_SIZE:
                    valid = False
                    break
                frames.append(frame)
            if valid:
                sequences.append(frames)
                labels.append(label_idx)

    if not sequences:
        raise RuntimeError("No valid dynamic sequences found.")

    x = np.asarray(sequences, dtype=np.float32)
    y = to_categorical(labels, num_classes=len(classes))
    return x, y, classes


def build_model(num_classes: int):
    model = Sequential(
        [
            LSTM(64, return_sequences=True, activation="relu", input_shape=(SEQUENCE_LENGTH, FRAME_SIZE)),
            Dropout(0.2),
            LSTM(128, return_sequences=True, activation="relu"),
            Dropout(0.2),
            LSTM(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train LSTM model for dynamic sign recognition.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/dynamic"), help="Folder containing class/sequence/*.npy")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    x, y, classes = load_dynamic_dataset(args.data_dir)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42, stratify=np.argmax(y, axis=1))

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(num_classes=len(classes))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
        ModelCheckpoint(args.model_dir / "dynamic_lstm.keras", monitor="val_loss", save_best_only=True, verbose=1),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    np.save(args.model_dir / "dynamic_labels.npy", np.array(classes, dtype=object))
    print("Saved dynamic model and labels.")


if __name__ == "__main__":
    main()
