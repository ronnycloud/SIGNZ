"""
SignVision - Keras Baseline Model Training
==========================================
Trains a baseline LSTM model on frame-by-frame MediaPipe keypoint sequences
collected in MP_Data. Automatically discovers all recorded actions and sequence counts.
"""

import os
import argparse
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from src.data.vocabulary import ACTION_IDS, get_display_name

DATA_PATH = os.path.join("MP_Data")
DEFAULT_ACTIONS = np.array(ACTION_IDS)
DEFAULT_SEQUENCE_LENGTH = 30


def discover_available_actions(data_path=DATA_PATH, min_sequences=1):
    """Scan data_path for actions that have recorded sequences."""
    if not os.path.isdir(data_path):
        return []

    valid_actions = []
    for entry in sorted(os.listdir(data_path)):
        action_dir = os.path.join(data_path, entry)
        if os.path.isdir(action_dir):
            # Check for sequence subdirectories containing .npy files
            seq_dirs = [
                d for d in os.listdir(action_dir)
                if os.path.isdir(os.path.join(action_dir, d)) and len(os.listdir(os.path.join(action_dir, d))) > 0
            ]
            if len(seq_dirs) >= min_sequences:
                valid_actions.append(entry)

    return valid_actions


def load_dataset(data_path=DATA_PATH, actions=None, sequence_length=DEFAULT_SEQUENCE_LENGTH):
    """Loads sequences for all available actions dynamically."""
    if actions is None or len(actions) == 0:
        actions = discover_available_actions(data_path)

    if not actions:
        raise ValueError(f"No actions with recorded sequences found in '{data_path}'. Please run 1_collect_data.py first.")

    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []

    print(f"[Dataset] Loading data for {len(actions)} actions from '{data_path}'...")
    for action in actions:
        action_dir = os.path.join(data_path, action)
        seq_dirs = sorted(
            [d for d in os.listdir(action_dir) if os.path.isdir(os.path.join(action_dir, d))],
            key=lambda x: int(x) if x.isdigit() else x,
        )

        for seq_name in seq_dirs:
            seq_path = os.path.join(action_dir, seq_name)
            frame_files = sorted(
                [f for f in os.listdir(seq_path) if f.endswith(".npy")],
                key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x,
            )

            if len(frame_files) < sequence_length:
                continue

            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(seq_path, f"{frame_num}.npy"))
                window.append(res)

            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences, dtype=np.float32)
    y = to_categorical(labels, num_classes=len(actions)).astype(int)

    print(f"[Dataset] Loaded {len(sequences)} total sequences across {len(actions)} classes.")
    test_size = 0.1 if len(sequences) >= 20 else 0.05
    return train_test_split(X, y, test_size=test_size, random_state=42), actions


def build_model(input_shape, num_classes):
    """Constructs multi-layer LSTM model for sequence classification."""
    model = Sequential([
        LSTM(64, return_sequences=True, activation="relu", input_shape=input_shape),
        Dropout(0.2),
        LSTM(128, return_sequences=True, activation="relu"),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    return model


def train(data_path=DATA_PATH, epochs=200, batch_size=16):
    """Train Keras LSTM on collected dataset."""
    (X_train, X_test, y_train, y_test), actions = load_dataset(data_path=data_path)

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(actions)

    print(f"[Model] Input shape: {input_shape}, Classes: {num_classes}")
    model = build_model(input_shape, num_classes)
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

    log_dir = os.path.join("Logs")
    os.makedirs(log_dir, exist_ok=True)
    tb_callback = TensorBoard(log_dir=log_dir)
    early_stop = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)

    print(f"[Train] Training for {epochs} epochs with batch size {batch_size}...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test) if len(X_test) > 0 else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[tb_callback, early_stop],
    )
    model.summary()

    # Evaluation
    if len(X_test) > 0:
        yhat = model.predict(X_test)
        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat_classes = np.argmax(yhat, axis=1).tolist()
        acc = accuracy_score(ytrue, yhat_classes)
        print(f"[Evaluation] Test Accuracy: {acc * 100:.2f}%")

    model.save("action.h5")
    # Save label map for 3_realtime_detection.py
    with open("action_labels.json", "w", encoding="utf-8") as f:
        json.dump(list(actions), f, indent=2)

    print("Model saved to 'action.h5' and labels to 'action_labels.json'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline LSTM model on MP_Data")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to landmark directory")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train(data_path=args.data_path, epochs=args.epochs, batch_size=args.batch_size)
