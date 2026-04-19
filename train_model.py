import os
import numpy as np
import librosa
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# CONFIG
# ==========================================
DATASET_PATH = "IRMAS-TrainingData/IRMAS-TrainingData"
IMG_SIZE = 128
SR = 22050
N_MELS = 128

# ==========================================
# LOAD DATA
# ==========================================
X = []
y = []

classes = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

print("Classes:", classes)

for label, folder in enumerate(classes):
    folder_path = os.path.join(DATASET_PATH, folder)

    for file in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(folder_path, file)

        try:
            audio, sr = librosa.load(file_path, sr=SR, duration=3)

            # MEL SPECTROGRAM
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=N_MELS
            )

            mel_db = librosa.power_to_db(mel, ref=np.max)

            # RESIZE
            mel_db = cv2.resize(mel_db, (IMG_SIZE, IMG_SIZE))

            # ✅ CORRECT NORMALIZATION
            mel_db = (mel_db + 80) / 80

            X.append(mel_db)
            y.append(label)

        except Exception:
            continue

# ==========================================
# PREPARE DATA
# ==========================================
X = np.array(X, dtype=np.float32)
y = np.array(y)

X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)

# ==========================================
# MODEL
# ==========================================
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(256, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),

    # ✅ FINAL OUTPUT
    layers.Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# TRAINING
# ==========================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        patience=3,
        factor=0.3
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=callbacks
)

# ==========================================
# EVALUATION
# ==========================================
val_loss, val_acc = model.evaluate(X_val, y_val)
print("Validation Accuracy:", val_acc)

# ==========================================
# SAVE MODEL
# ==========================================
model.save("multilabel_instrument_model_FINAL.h5")
print("✅ Model saved successfully!")