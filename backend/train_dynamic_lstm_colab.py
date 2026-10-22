from google.colab import drive
drive.mount("/content/drive", force_remount=True)

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

DATA_PATH = "/content/MP_Data"
SEQUENCE_LENGTH = 30
FRAME_SIZE = 1662
MODEL_PATH = "/content/dynamic_lstm.keras"
LABELS_PATH = "/content/dynamic_labels.npy"
DRIVE_EXPORT_DIR = "/content/drive/MyDrive/signlang_models"

print("DATA_PATH exists:", os.path.exists(DATA_PATH))
if os.path.exists(DATA_PATH):
    available = sorted(
        [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    )
    print("Class folders:", len(available))
    print("Sample classes:", available[:10])

classes = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
if not classes:
    raise RuntimeError("No class folders found in DATA_PATH.")

X, y = [], []
bad_sequences = 0

for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(DATA_PATH, class_name)
    seq_dirs = sorted([d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))])

    for seq_id in seq_dirs:
        seq_dir = os.path.join(class_dir, seq_id)
        frames = []
        valid = True

        for frame_id in range(SEQUENCE_LENGTH):
            frame_file = os.path.join(seq_dir, f"{frame_id}.npy")
            if not os.path.exists(frame_file):
                valid = False
                break

            frame = np.load(frame_file)
            if frame.shape[0] != FRAME_SIZE:
                valid = False
                break
            frames.append(frame)

        if valid:
            X.append(frames)
            y.append(class_idx)
        else:
            bad_sequences += 1

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)

print("Loaded X shape:", X.shape)
print("Loaded y shape:", y.shape)
print("Bad sequences skipped:", bad_sequences)
print("Total classes:", len(classes))

if X.shape[0] == 0:
    raise RuntimeError("No valid sequences loaded. Check DATA_PATH structure.")

y_cat = to_categorical(y, num_classes=len(classes))

X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.15, random_state=42, stratify=y
)

print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)

model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, FRAME_SIZE)),
    LSTM(64, return_sequences=True, activation="tanh"),
    Dropout(0.2),
    LSTM(128, return_sequences=True, activation="tanh"),
    Dropout(0.2),
    LSTM(64, activation="tanh"),
    BatchNormalization(),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(len(classes), activation="softmax"),
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
]

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=16,
    callbacks=callbacks,
)

np.save(LABELS_PATH, np.array(classes, dtype=object))
print("Saved model :", MODEL_PATH)
print("Saved labels:", LABELS_PATH)

os.makedirs(DRIVE_EXPORT_DIR, exist_ok=True)
os.system(f'cp "{MODEL_PATH}" "{DRIVE_EXPORT_DIR}/dynamic_lstm.keras"')
os.system(f'cp "{LABELS_PATH}" "{DRIVE_EXPORT_DIR}/dynamic_labels.npy"')
print("Copied to Drive:", DRIVE_EXPORT_DIR)
