from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np

import shutil

if os.path.exists("/content/drive"):
    shutil.rmtree("/content/drive")

from google.colab import drive
drive.mount("/content/drive")

!ls /content/drive/MyDrive

PROJECT_PATH  = '/content/drive/MyDrive/SignLanguageProject'
DATA_PATH     = f'{PROJECT_PATH}/MP_Data'
MODELS_PATH   = f'{PROJECT_PATH}/models'
DATASET_PATH  = f'{PROJECT_PATH}/datasets/sign-language-gesture-images-dataset/Gesture Image Data'

for p in [DATA_PATH, MODELS_PATH]:
    os.makedirs(p, exist_ok=True)

print("Drive mounted")
print(f"Dataset path : {DATASET_PATH}")
print(f"Data path    : {DATA_PATH}")
print(f"Models path  : {MODELS_PATH}")

# Confirm dataset folder exists
if os.path.exists(DATASET_PATH):
    # Sort folders case-insensitively
    folders = sorted(os.listdir(DATASET_PATH), key=str.lower)
    print(f"\n Dataset found — {len(folders)} class folders")
    print(folders)
else:
    print("Dataset folder not found — check your path above")

!pip install mediapipe opencv-python-headless -q

import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import os

print("TensorFlow :", tf.__version__)
print("GPU        :", tf.config.list_physical_devices('GPU'))

if '_' in folders:
    folders.remove('_')
    folders.insert(0, '_') # Insert at the beginning
actions = np.array(folders)

print(f"Total classes      : {len(actions)}")
print(f"  From dataset     : {len(folders)}")
print(f"\nAll actions:\n{actions}")

# Save immediately — needed for web app later
np.save(f'{MODELS_PATH}/actions.npy', actions)
print("\n actions.npy saved")

!pip uninstall mediapipe -y
!pip install mediapipe==0.10.20

import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

print("Holistic loaded successfully ")

def mediapipe_detection(image, model):
    image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Removing image.flags.writeable = False as it seems to cause issues with MediaPipe's internal handling
    results  = model.process(image)
    # Removing image.flags.writeable = True
    image    = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[r.x, r.y, r.z, r.visibility]
                     for r in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[r.x, r.y, r.z]
                     for r in results.face_landmarks.landmark]).flatten() \
           if results.face_landmarks else np.zeros(468*3)
    lh   = np.array([[r.x, r.y, r.z]
                     for r in results.left_hand_landmarks.landmark]).flatten() \
           if results.left_hand_landmarks else np.zeros(21*3)
    rh   = np.array([[r.x, r.y, r.z]
                     for r in results.right_hand_landmarks.landmark]).flatten() \
           if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])   # → 1662 values

print(" Helper functions ready")

# from tqdm.notebook import tqdm

# SEQUENCE_LENGTH = 30    # frames per sequence
# MAX_SEQUENCES   = 50    # max sequences to create per class

# def convert_images_to_npy(dataset_path, output_path,
#                            sequence_length=30,
#                            max_sequences=50):
#     """
#     For each class folder:
#       1. Load all images
#       2. Run MediaPipe on each image to get 1662 keypoints
#       3. Group into sequences of 30 frames
#          - If class has 300+ images → use different images per frame
#          - If class has < 30 images → repeat images to fill 30 frames
#       4. Save each sequence as 30 individual .npy files
#     """
#     all_classes = sorted(os.listdir(dataset_path))
#     print(f"Classes found: {len(all_classes)}\n")

#     total_sequences = 0
#     total_failed    = 0

#     for class_name in tqdm(all_classes, desc="Converting classes"):
#         class_path = os.path.join(dataset_path, class_name)

#         if not os.path.isdir(class_path):
#             continue

#         # Get all image files
#         images = sorted([
#             f for f in os.listdir(class_path)
#             if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))
#         ])

#         if len(images) == 0:
#             print(f"  {class_name} — no images found, skipping")
#             continue

#         print(f"\n  {class_name} — {len(images)} images → "
#               f"creating sequences...")

#         # ── Extract keypoints from ALL images first ────────────
#         all_keypoints = []
#         failed        = 0

#         with mp_holistic.Holistic(
#                 static_image_mode=True,
#                 min_detection_confidence=0.3,
#                 min_tracking_confidence=0.3) as holistic:

#             for img_file in images:
#                 img_path = os.path.join(class_path, img_file)
#                 img      = cv2.imread(img_path)

#                 if img is None:
#                     all_keypoints.append(np.zeros(1662))
#                     failed += 1
#                     continue

#                 # Resize for faster processing
#                 img = cv2.resize(img, (640, 480))
#                 _, results = mediapipe_detection(img, holistic)
#                 kp = extract_keypoints(results)

#                 # If no hand detected, try flipping the image
#                 if np.sum(np.abs(kp)) == 0:
#                     img_flipped = cv2.flip(img, 1)
#                     _, results  = mediapipe_detection(
#                         img_flipped, holistic)
#                     kp = extract_keypoints(results)

#                 if np.sum(np.abs(kp)) == 0:
#                     failed += 1

#                 all_keypoints.append(kp)

#         print(f"    Keypoints extracted: {len(all_keypoints)} "
#               f"| Failed: {failed}")

#         # ── Build sequences from extracted keypoints ───────────
#         seq_id = 0

#         if len(all_keypoints) >= sequence_length:
#             # Slide a window across all keypoints
#             step = max(1, len(all_keypoints) //
#                        min(max_sequences,
#                            len(all_keypoints) // sequence_length))
#             i = 0
#             while (i + sequence_length <= len(all_keypoints)
#                    and seq_id < max_sequences):
#                 sequence = all_keypoints[i : i + sequence_length]
#                 _save_sequence(output_path, class_name,
#                                seq_id, sequence)
#                 seq_id += 1
#                 i      += step
#         else:
#             # Not enough images — repeat to fill sequence_length
#             for s in range(min(max_sequences, 30)):
#                 sequence = []
#                 for f in range(sequence_length):
#                     idx = f % len(all_keypoints)
#                     sequence.append(all_keypoints[idx])
#                 _save_sequence(output_path, class_name,
#                                seq_id, sequence)
#                 seq_id += 1

#         print(f"  {seq_id} sequences saved "
#               f"| {failed} failed frames")
#         total_sequences += seq_id
#         total_failed    += failed

#     print(f"\n{'='*50}")
#     print(f"   Conversion complete!")
#     print(f"   Total sequences : {total_sequences}")
#     print(f"   Total failed    : {total_failed}")
#     print(f"{'='*50}")


# def _save_sequence(output_path, class_name, seq_id, sequence):
#     """Save a list of keypoint arrays as numbered .npy files."""
#     out_dir = os.path.join(output_path, class_name, str(seq_id))
#     os.makedirs(out_dir, exist_ok=True)
#     for frame_num, kp in enumerate(sequence):
#         np.save(os.path.join(out_dir, str(frame_num)), kp)


# # ── Run the conversion ─────────────────────────────────────────
# # This will take 1-2 hours depending on dataset size
# # Safe to interrupt and restart — skips already-converted classes

# convert_images_to_npy(
#     dataset_path=DATASET_PATH,
#     output_path=DATA_PATH,
#     sequence_length=SEQUENCE_LENGTH,
#     max_sequences=MAX_SEQUENCES
# )

# from tqdm.notebook import tqdm
# import shutil
# import cv2

# def reprocess_problem_classes(problem_classes, dataset_path,
#                                output_path, sequence_length=30,
#                                max_sequences=50):
#     """
#     Re-runs conversion on specific classes with lower confidence
#     thresholds and more aggressive preprocessing to catch more hands.
#     Overwrites existing sequences for these classes.
#     """
#     print(f"Re-processing: {problem_classes}\n")

#     for class_name in problem_classes:
#         class_path = os.path.join(dataset_path, class_name)

#         if not os.path.isdir(class_path):
#             print(f" {class_name} not found, skipping")
#             continue

#         images = sorted([
#             f for f in os.listdir(class_path)
#             if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))
#         ])

#         print(f"\n{class_name} — {len(images)} images")
#         all_keypoints = []
#         failed        = 0

#         with mp_holistic.Holistic(
#                 static_image_mode=True,
#                 min_detection_confidence=0.1,  # ← much lower
#                 min_tracking_confidence=0.1,
#                 model_complexity=1) as holistic:

#             for img_file in tqdm(images,
#                                   desc=f"  {class_name}",
#                                   leave=False):
#                 img_path = os.path.join(class_path, img_file)
#                 img      = cv2.imread(img_path)
#                 if img is None:
#                     all_keypoints.append(np.zeros(1662))
#                     failed += 1
#                     continue

#                 # Try multiple preprocessing approaches
#                 kp = np.zeros(1662)

#                 # Attempt 1 — original resized
#                 img_resized = cv2.resize(img, (640, 480))
#                 _, results  = mediapipe_detection(img_resized, holistic)
#                 kp = extract_keypoints(results)

#                 # Attempt 2 — flip horizontally
#                 if np.sum(np.abs(kp)) == 0:
#                     img_flip   = cv2.flip(img_resized, 1)
#                     _, results = mediapipe_detection(img_flip, holistic)
#                     kp = extract_keypoints(results)

#                 # Attempt 3 — enhance contrast
#                 if np.sum(np.abs(kp)) == 0:
#                     img_gray = cv2.cvtColor(img_resized,
#                                             cv2.COLOR_BGR2GRAY)
#                     img_eq   = cv2.equalizeHist(img_gray)
#                     img_eq3  = cv2.cvtColor(img_eq,
#                                              cv2.COLOR_GRAY2BGR)
#                     _, results = mediapipe_detection(img_eq3, holistic)
#                     kp = extract_keypoints(results)

#                 # Attempt 4 — larger resolution
#                 if np.sum(np.abs(kp)) == 0:
#                     img_large  = cv2.resize(img, (1280, 960))
#                     _, results = mediapipe_detection(img_large, holistic)
#                     kp = extract_keypoints(results)

#                 if np.sum(np.abs(kp)) == 0:
#                     failed += 1

#                 all_keypoints.append(kp)

#         failure_rate = failed / len(images) * 100
#         print(f"  Failed: {failed}/{len(images)} "
#               f"({failure_rate:.1f}%)")

#         # Remove old sequences for this class

#         old_path = os.path.join(output_path, class_name)
#         if os.path.exists(old_path):
#             shutil.rmtree(old_path)
#             print(f"  Old sequences deleted")

#         # Save new sequences
#         seq_id = 0
#         if len(all_keypoints) >= sequence_length:
#             step = max(1, len(all_keypoints) //
#                        min(max_sequences,
#                            len(all_keypoints) // sequence_length))
#             i = 0
#             while (i + sequence_length <= len(all_keypoints)
#                    and seq_id < max_sequences):
#                 sequence = all_keypoints[i : i + sequence_length]
#                 _save_sequence(output_path, class_name,
#                                seq_id, sequence)
#                 seq_id += 1
#                 i      += step

#         print(f" {seq_id} sequences saved")

#     print("\nReprocessing complete!")


# # Run on the high-failure classes
# reprocess_problem_classes(
#     problem_classes=['4', '6', '7', '3', '5','9','O','P','hungry','ready','yes'],
#     dataset_path=DATASET_PATH,
#     output_path=DATA_PATH
# )

!pip install mediapipe==0.10.20 opencv-python numpy tqdm

# import os
# import cv2
# import numpy as np
# import mediapipe as mp
# from tqdm.notebook import tqdm


# # ===============================
# # MediaPipe setup
# # ===============================

# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils


# # ===============================
# # MediaPipe detection function
# # (Python 3.12 compatibility fix)
# # ===============================

# def mediapipe_detection(image, model):

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # REQUIRED FIX for Colab Python 3.12
#     image = np.ascontiguousarray(image)

#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True

#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     return image, results


# # ===============================
# # Extract 1662 keypoints
# # ===============================

# def extract_keypoints(results):

#     pose = np.array([
#         [r.x, r.y, r.z, r.visibility]
#         for r in results.pose_landmarks.landmark
#     ]).flatten() if results.pose_landmarks else np.zeros(33 * 4)


#     face = np.array([
#         [r.x, r.y, r.z]
#         for r in results.face_landmarks.landmark
#     ]).flatten() if results.face_landmarks else np.zeros(468 * 3)


#     lh = np.array([
#         [r.x, r.y, r.z]
#         for r in results.left_hand_landmarks.landmark
#     ]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)


#     rh = np.array([
#         [r.x, r.y, r.z]
#         for r in results.right_hand_landmarks.landmark
#     ]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)


#     return np.concatenate([pose, face, lh, rh])


# # ===============================
# # Save sequence as .npy files
# # ===============================

# def _save_sequence(output_path, class_name, seq_id, sequence):

#     out_dir = os.path.join(output_path, class_name, str(seq_id))
#     os.makedirs(out_dir, exist_ok=True)

#     for frame_num, kp in enumerate(sequence):

#         np.save(
#             os.path.join(out_dir, str(frame_num)),
#             kp
#         )


# # ===============================
# # Dataset conversion pipeline
# # ===============================

# SEQUENCE_LENGTH = 30
# MAX_SEQUENCES = 50


# def convert_images_to_npy(
#         dataset_path,
#         output_path,
#         sequence_length=30,
#         max_sequences=50):

#     all_classes = sorted(os.listdir(dataset_path))

#     print(f"Classes found: {len(all_classes)}\n")

#     total_sequences = 0
#     total_failed = 0


#     for class_name in tqdm(all_classes, desc="Processing classes"):

#         class_path = os.path.join(dataset_path, class_name)

#         if not os.path.isdir(class_path):
#             continue


#         images = sorted([
#             f for f in os.listdir(class_path)
#             if f.lower().endswith(
#                 ('.jpg', '.jpeg', '.png', '.bmp')
#             )
#         ])


#         if len(images) == 0:

#             print(f"{class_name} skipped (no images)")
#             continue


#         print(f"\n{class_name} → {len(images)} images")


#         all_keypoints = []
#         failed = 0


#         with mp_holistic.Holistic(

#                 static_image_mode=True,
#                 model_complexity=1,
#                 min_detection_confidence=0.3,
#                 min_tracking_confidence=0.3

#         ) as holistic:


#             for img_file in tqdm(
#                     images,
#                     desc=f"{class_name}",
#                     leave=False):


#                 img_path = os.path.join(
#                     class_path,
#                     img_file
#                 )


#                 img = cv2.imread(img_path)


#                 if img is None:

#                     all_keypoints.append(
#                         np.zeros(1662)
#                     )

#                     failed += 1
#                     continue


#                 img = cv2.resize(
#                     img,
#                     (640, 480)
#                 ).copy()


#                 _, results = mediapipe_detection(
#                     img,
#                     holistic
#                 )


#                 kp = extract_keypoints(results)


#                 # Try flipped version if detection failed
#                 if not np.any(kp):

#                     img_flip = cv2.flip(
#                         img,
#                         1
#                     ).copy()


#                     _, results = mediapipe_detection(
#                         img_flip,
#                         holistic
#                     )


#                     kp = extract_keypoints(results)


#                 if not np.any(kp):

#                     failed += 1


#                 all_keypoints.append(kp)


#         print(
#             f"Keypoints extracted: {len(all_keypoints)}"
#             f" | Failed: {failed}"
#         )


#         seq_id = 0


#         if len(all_keypoints) >= sequence_length:


#             step = max(
#                 1,
#                 len(all_keypoints) //
#                 min(
#                     max_sequences,
#                     len(all_keypoints) //
#                     sequence_length
#                 )
#             )


#             i = 0


#             while (
#                     i + sequence_length
#                     <= len(all_keypoints)
#                     and seq_id < max_sequences
#             ):


#                 sequence = all_keypoints[
#                     i:i + sequence_length
#                 ]


#                 _save_sequence(
#                     output_path,
#                     class_name,
#                     seq_id,
#                     sequence
#                 )


#                 seq_id += 1
#                 i += step


#         else:


#             for s in range(
#                     min(max_sequences, 30)
#             ):


#                 sequence = []


#                 for f in range(sequence_length):

#                     idx = f % len(all_keypoints)

#                     sequence.append(
#                         all_keypoints[idx]
#                     )


#                 _save_sequence(
#                     output_path,
#                     class_name,
#                     seq_id,
#                     sequence
#                 )


#                 seq_id += 1


#         print(
#             f"{seq_id} sequences saved"
#             f" | {failed} failed frames"
#         )


#         total_sequences += seq_id
#         total_failed += failed


#     print("\n===============================")
#     print("Dataset conversion complete")
#     print(f"Total sequences: {total_sequences}")
#     print(f"Total failed frames: {total_failed}")
#     print("===============================")


# # ===============================
# # RUN PIPELINE
# # ===============================

# convert_images_to_npy(
#     dataset_path=DATASET_PATH,
#     output_path=DATA_PATH,
#     sequence_length=SEQUENCE_LENGTH,
#     max_sequences=MAX_SEQUENCES
# )

!pip uninstall mediapipe tensorflow keras -y

!pip install mediapipe==0.10.20 opencv-python numpy tqdm

# import mediapipe as mp

# print("MediaPipe version:", mp.__version__)

# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# print("Holistic loaded successfully ")

# import os
# import cv2
# import numpy as np
# import shutil
# from tqdm.notebook import tqdm


# def reprocess_problem_classes(problem_classes,
#                               dataset_path,
#                               output_path,
#                               sequence_length=30,
#                               max_sequences=50):

#     print(f"Re-processing: {problem_classes}\n")

#     for class_name in problem_classes:

#         class_path = os.path.join(dataset_path, class_name)

#         if not os.path.isdir(class_path):

#             print(f"{class_name} not found — skipping")
#             continue


#         images = sorted([
#             f for f in os.listdir(class_path)
#             if f.lower().endswith(
#                 ('.jpg','.jpeg','.png','.bmp')
#             )
#         ])


#         print(f"\n{class_name} — {len(images)} images")


#         all_keypoints = []
#         failed = 0


#         with mp_holistic.Holistic(

#                 static_image_mode=True,
#                 model_complexity=1,
#                 min_detection_confidence=0.1,
#                 min_tracking_confidence=0.1

#         ) as holistic:


#             for img_file in tqdm(images,
#                                  desc=f"{class_name}",
#                                  leave=False):


#                 img_path = os.path.join(
#                     class_path,
#                     img_file
#                 )


#                 img = cv2.imread(img_path)


#                 if img is None:

#                     all_keypoints.append(
#                         np.zeros(1662)
#                     )

#                     failed += 1
#                     continue


#                 img = cv2.resize(
#                     img,
#                     (640,480)
#                 ).copy()


#                 _, results = mediapipe_detection(
#                     img,
#                     holistic
#                 )


#                 kp = extract_keypoints(results)


#                 # try flipped image
#                 if not np.any(kp):

#                     img_flip = cv2.flip(
#                         img,
#                         1
#                     ).copy()


#                     _, results = mediapipe_detection(
#                         img_flip,
#                         holistic
#                     )


#                     kp = extract_keypoints(results)


#                 if not np.any(kp):

#                     failed += 1


#                 all_keypoints.append(kp)


#         print(
#             f"Failed detections: {failed}/{len(images)}"
#         )


#         # delete old sequences

#         old_path = os.path.join(
#             output_path,
#             class_name
#         )


#         if os.path.exists(old_path):

#             shutil.rmtree(old_path)

#             print("Old sequences deleted")


#         seq_id = 0


#         if len(all_keypoints) >= sequence_length:


#             step = max(
#                 1,
#                 len(all_keypoints) //
#                 min(
#                     max_sequences,
#                     len(all_keypoints)//sequence_length
#                 )
#             )


#             i = 0


#             while (
#                     i + sequence_length <= len(all_keypoints)
#                     and seq_id < max_sequences
#             ):


#                 sequence = all_keypoints[
#                     i:i+sequence_length
#                 ]


#                 _save_sequence(
#                     output_path,
#                     class_name,
#                     seq_id,
#                     sequence
#                 )


#                 seq_id += 1
#                 i += step


#         print(f"{seq_id} sequences saved")


#     print("\nReprocessing complete ")

import cv2
import numpy as np
import mediapipe as mp

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ===============================
# MediaPipe detection function
# ===============================

def mediapipe_detection(image, model):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Required fix for Colab Python 3.12
    image = np.ascontiguousarray(image)

    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results


# ===============================
# Extract 1662 keypoints
# ===============================

def extract_keypoints(results):

    pose = np.array([
        [r.x, r.y, r.z, r.visibility]
        for r in results.pose_landmarks.landmark
    ]).flatten() if results.pose_landmarks else np.zeros(33*4)


    face = np.array([
        [r.x, r.y, r.z]
        for r in results.face_landmarks.landmark
    ]).flatten() if results.face_landmarks else np.zeros(468*3)


    lh = np.array([
        [r.x, r.y, r.z]
        for r in results.left_hand_landmarks.landmark
    ]).flatten() if results.left_hand_landmarks else np.zeros(21*3)


    rh = np.array([
        [r.x, r.y, r.z]
        for r in results.right_hand_landmarks.landmark
    ]).flatten() if results.right_hand_landmarks else np.zeros(21*3)


    return np.concatenate([pose, face, lh, rh])


# ===============================
# Sequence saving function
# ===============================

import os

def _save_sequence(output_path, class_name, seq_id, sequence):

    out_dir = os.path.join(output_path, class_name, str(seq_id))
    os.makedirs(out_dir, exist_ok=True)

    for frame_num, kp in enumerate(sequence):

        np.save(
            os.path.join(out_dir, str(frame_num)),
            kp
        )

print("Helper functions ready ")

reprocess_problem_classes(
    problem_classes=[
        '4','6','7','3','5','9',
        'O','P',
        'hungry','ready','yes'
    ],
    dataset_path=DATASET_PATH,
    output_path=DATA_PATH
)

import os
import numpy as np
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

# Install tensorflow if not already installed
try:
    import tensorflow
except ImportError:
    !pip install tensorflow==2.19.0 # Install tensorflow and its compatible keras version
    import tensorflow # Re-import after installation

from tensorflow.keras.utils import to_categorical

# Load class labels exactly as saved earlier
actions = np.load(f"{MODELS_PATH}/actions.npy")

sequence_length = 30

sequences = []
labels = []

print("Loading dataset...\n")

for action in tqdm(actions):

    action_path = os.path.join(DATA_PATH, action)

    if not os.path.isdir(action_path):
        continue

    for sequence in os.listdir(action_path):

        window = []

        sequence_path = os.path.join(action_path, sequence)

        for frame_num in range(sequence_length):

            frame_path = os.path.join(
                sequence_path,
                f"{frame_num}.npy"
            )

            res = np.load(frame_path)

            window.append(res)

        sequences.append(window)

        labels.append(
            np.where(actions == action)[0][0]
        )


X = np.array(sequences)
y = to_categorical(labels)

print("\nDataset loaded successfully ")
print("X shape:", X.shape)
print("y shape:", y.shape)

import os
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.utils import to_categorical

actions = np.load(f"{MODELS_PATH}/actions.npy")

sequence_length = 30

sequences = []
labels = []

bad_files = 0

print("Loading dataset...\n")

for idx, action in enumerate(tqdm(actions)):

    action_path = os.path.join(DATA_PATH, action)

    if not os.path.isdir(action_path):
        continue

    for sequence in sorted(os.listdir(action_path)):

        sequence_path = os.path.join(action_path, sequence)

        window = []
        corrupted_sequence = False

        for frame_num in range(sequence_length):

            frame_path = os.path.join(
                sequence_path,
                f"{frame_num}.npy"
            )

            try:
                res = np.load(frame_path)
                window.append(res)

            except:
                corrupted_sequence = True
                bad_files += 1
                break


        if not corrupted_sequence:

            sequences.append(window)
            labels.append(idx)


X = np.array(sequences, dtype=np.float32)
y = to_categorical(labels)

print("\nDataset loaded successfully")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Corrupted sequences skipped:", bad_files)


# 🔥 SAVE immediately so runtime crashes don't matter

np.save(f"{MODELS_PATH}/X.npy", X)
np.save(f"{MODELS_PATH}/y.npy", y)

print("\nSaved dataset to Drive ")

import numpy as np

X = np.load(f"{MODELS_PATH}/X.npy")
y = np.load(f"{MODELS_PATH}/y.npy")
X = X / np.max(np.abs(X))
actions = np.load(f"{MODELS_PATH}/actions.npy")

print("Dataset restored successfully ")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Total classes:", len(actions))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

# Load actions from Drive (same mapping used during preprocessing)
actions = np.load(f"{MODELS_PATH}/actions.npy")

print("Total classes:", len(actions))


# Build model
model = Sequential()

model.add(
    LSTM(
        64,
        return_sequences=True,
        activation='relu',
        input_shape=(30, 1662)
    )
)

model.add(Dropout(0.2))


model.add(
    LSTM(
        128,
        return_sequences=True,
        activation='relu'
    )
)

model.add(Dropout(0.2))


model.add(
    LSTM(
        64,
        return_sequences=False,
        activation='relu'
    )
)
model.add(BatchNormalization())


model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))


# Output layer matches number of gesture classes
model.add(
    Dense(len(actions), activation='softmax')
)


model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=12,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    f"{MODELS_PATH}/best_sign_language_model.h5",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=5,
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=80,
    batch_size=16,
    callbacks=[early_stop, checkpoint, lr_scheduler]
)

from tensorflow.keras.models import load_model

model = load_model(f"{MODELS_PATH}/best_sign_language_model.h5")

loss, accuracy = model.evaluate(X_test, y_test)

print("Final Test Accuracy:", accuracy)



model.save(f"{MODELS_PATH}/sign_language_model.keras")

print("Model saved successfully ")

print("actions:", len(actions))
print("X shape:", X.shape)
print("y shape:", y.shape)
print("model output:", model.output_shape)