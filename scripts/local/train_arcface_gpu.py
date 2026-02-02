#!/usr/bin/env python
"""
GPU Optimized ArcFace Training
Includes Mixed Precision and Memory Growth settings.
"""

import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Enable Mixed Precision (FP16)
# Speeds up training on RTX GPUs and reduces memory usage
from tensorflow.keras import mixed_precision

try:
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)
    print("Mixed Precision enabled: mixed_float16")
except Exception as e:
    print(f"Failed to set mixed precision: {e}")

# 2. Configure GPU Memory Growth
# Prevents TensorFlow from allocating all GPU memory at once
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Memory Growth enabled for {len(gpus)} GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("WARNING: No GPU detected. Training will be slow on CPU.")

# Configuration
RAW_DIR = PROJECT_ROOT / "data" / "images" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
IMG_SIZE = 224
EMBEDDING_DIM = 128
# 3. Increased Batch Size for GPU
# CPU was 16, GPU can handle 32 or 64 easily
BATCH_SIZE = 32
EPOCHS = 10
ARCFACE_SCALE = 30.0
ARCFACE_MARGIN = 0.5


class ArcFaceLayer(layers.Layer):
    """ArcFace loss layer for metric learning."""

    def __init__(self, num_classes, embedding_dim, scale=30.0, margin=0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.margin = margin

    def build(self, input_shape):
        self.W = self.add_weight(
            name="arcface_weights",
            shape=(self.embedding_dim, self.num_classes),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs, labels=None, training=None):
        # Cast inputs to float32 for stability in mixed precision
        inputs = tf.cast(inputs, tf.float32)

        # L2 normalize
        embeddings = tf.nn.l2_normalize(inputs, axis=1)
        weights = tf.nn.l2_normalize(self.W, axis=0)

        # Cosine similarity
        cos_theta = tf.matmul(embeddings, weights)
        cos_theta = tf.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        if training and labels is not None:
            theta = tf.acos(cos_theta)
            one_hot = tf.one_hot(labels, self.num_classes)

            theta_m = theta + one_hot * self.margin
            cos_theta_m = tf.cos(theta_m)

            logits = self.scale * cos_theta_m
        else:
            logits = self.scale * cos_theta

        return logits


def load_dataset():
    """Load images and labels."""
    print("Loading dataset...")
    images = []
    labels = []
    label_names = []

    part_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    for part_idx, part_dir in enumerate(part_dirs):
        label_names.append(part_dir.name)
        for color_dir in part_dir.iterdir():
            if not color_dir.is_dir():
                continue
            for img_path in color_dir.glob("*.jpg"):
                images.append(str(img_path))
                labels.append(part_idx)

    return images, labels, label_names


def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def create_dataset(image_paths, labels, training=True):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def load_and_preprocess(path, label):
        return preprocess_image(path), label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(1000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def create_embedding_model():
    base_model = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg"
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(EMBEDDING_DIM, name="embeddings")(x)

    # UnitNormalization is mostly stable in fp16, but safe to keep as is
    embeddings = layers.UnitNormalization(axis=1)(x)

    # Cast to float32 for output if using mixed precision (embeddings usually better in fp32)
    embeddings = layers.Activation("linear", dtype="float32")(embeddings)

    return Model(inputs, embeddings, name="embedding_model")


class ArcFaceTrainer(keras.Model):
    def __init__(self, embedding_model, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.embedding_model = embedding_model
        # Ensure ArcFace calculations are in float32
        self.arcface = ArcFaceLayer(num_classes, EMBEDDING_DIM, ARCFACE_SCALE, ARCFACE_MARGIN, dtype="float32")
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.accuracy_tracker = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    def call(self, inputs, training=None):
        return self.embedding_model(inputs, training=training)

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            embeddings = self.embedding_model(images, training=True)
            logits = self.arcface(embeddings, labels, training=True)
            loss = keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

            # Scale loss for mixed precision
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(scaled_loss, self.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(labels, logits)
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_tracker.result()}

    def test_step(self, data):
        images, labels = data
        embeddings = self.embedding_model(images, training=False)
        logits = self.arcface(embeddings, labels, training=False)
        loss = keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(labels, logits)
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_tracker.result()}


def main():
    print("=" * 60)
    print("ArcFace Training - GPU Optimized")
    print("=" * 60)

    image_paths, labels, label_names = load_dataset()
    num_classes = len(label_names)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    print(f"Training on {len(train_paths)} images, validating on {len(val_paths)}")

    train_ds = create_dataset(train_paths, train_labels, training=True)
    val_ds = create_dataset(val_paths, val_labels, training=False)

    embedding_model = create_embedding_model()
    trainer = ArcFaceTrainer(embedding_model, num_classes)

    # Use Mixed Precision optimizer wrapper implicitly handled by set_global_policy?
    # Actually, in newer TF, just passing optimizer is enough if policy is set.
    trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

    print("\nStarting training...")
    trainer.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

    # Save optimized model
    out_path = MODEL_DIR / "lego_embedding_model_gpu.keras"
    embedding_model.save(out_path)
    print(f"\nSaved model: {out_path}")


if __name__ == "__main__":
    main()
