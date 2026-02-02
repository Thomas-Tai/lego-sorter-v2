#!/usr/bin/env python
"""
Quick Training Test Script
Tests if the captured LEGO images can train an AI model.
Uses EfficientNetB0 with transfer learning.
"""

import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Configuration
RAW_DIR = PROJECT_ROOT / "data" / "images" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "images" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
IMG_SIZE = 224  # EfficientNetB0 input size
BATCH_SIZE = 16
EPOCHS = 5


def load_dataset():
    """Load images and labels from raw directory."""
    print("Loading dataset...")

    images = []
    labels = []
    label_names = []

    # Scan all part directories
    part_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])

    for part_idx, part_dir in enumerate(part_dirs):
        part_id = part_dir.name
        label_names.append(part_id)

        # Get all images for this part
        for color_dir in part_dir.iterdir():
            if not color_dir.is_dir():
                continue
            for img_path in color_dir.glob("*.jpg"):
                images.append(str(img_path))
                labels.append(part_idx)

    print(f"Found {len(images)} images across {len(label_names)} parts")
    return images, labels, label_names


def preprocess_image(image_path):
    """Load and preprocess a single image."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def create_dataset(image_paths, labels, training=True):
    """Create tf.data.Dataset from paths and labels."""
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def load_and_preprocess(path, label):
        img = preprocess_image(path)
        return img, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(1000)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def create_model(num_classes):
    """Create EfficientNetB0-based classifier."""
    base_model = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg"
    )

    # Freeze base model
    base_model.trainable = False

    model = keras.Sequential(
        [
            base_model,
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_training_history(history, output_path):
    """Save training history plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history.history["loss"], label="Train")
    ax1.plot(history.history["val_loss"], label="Validation")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # Accuracy
    ax2.plot(history.history["accuracy"], label="Train")
    ax2.plot(history.history["val_accuracy"], label="Validation")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Training plot saved: {output_path}")


def main():
    print("=" * 60)
    print("LEGO Part Classification - Quick Training Test")
    print("=" * 60)

    # Check GPU
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(gpus) > 0}")
    if gpus:
        print(f"  GPU: {gpus[0].name}")

    # Load data
    image_paths, labels, label_names = load_dataset()
    num_classes = len(label_names)

    if len(image_paths) < 20:
        print("ERROR: Not enough images for training (need >= 20)")
        return

    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    print(f"\nDataset split:")
    print(f"  Training: {len(train_paths)} images")
    print(f"  Validation: {len(val_paths)} images")
    print(f"  Classes: {num_classes}")

    # Create datasets
    train_ds = create_dataset(train_paths, train_labels, training=True)
    val_ds = create_dataset(val_paths, val_labels, training=False)

    # Create model
    print("\nCreating model...")
    model = create_model(num_classes)
    model.summary()

    # Train
    print(f"\nTraining for {EPOCHS} epochs...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

    # Results
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)

    final_loss = history.history["loss"][-1]
    final_acc = history.history["accuracy"][-1]
    final_val_loss = history.history["val_loss"][-1]
    final_val_acc = history.history["val_accuracy"][-1]

    print(f"Final Training Loss: {final_loss:.4f}")
    print(f"Final Training Accuracy: {final_acc:.2%}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.2%}")

    # Check if learning occurred
    loss_decreased = history.history["loss"][0] > history.history["loss"][-1]
    acc_increased = history.history["accuracy"][0] < history.history["accuracy"][-1]

    print("\n" + "=" * 60)
    if loss_decreased and acc_increased:
        print("[OK] Model is learning! Loss decreased, accuracy increased.")
        print("Dataset is VALID for training.")
    else:
        print("[WARN] Model may not be learning effectively.")
        print("Consider: more data, different augmentation, or hyperparameters.")
    print("=" * 60)

    # Save plot
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = MODEL_DIR / "training_history.png"
    plot_training_history(history, plot_path)

    # Save model
    model_path = MODEL_DIR / "lego_classifier_test.keras"
    model.save(model_path)
    print(f"Model saved: {model_path}")


if __name__ == "__main__":
    main()
