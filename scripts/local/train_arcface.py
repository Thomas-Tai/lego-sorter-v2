#!/usr/bin/env python
"""
Metric Learning Training with ArcFace Loss
Trains an embedding model that outputs feature vectors instead of class labels.
This allows adding new parts without retraining.
"""
import os
import sys
from pathlib import Path
import math

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
RAW_DIR = PROJECT_ROOT / "data" / "images" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
IMG_SIZE = 224
EMBEDDING_DIM = 128
BATCH_SIZE = 16
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
            name='arcface_weights',
            shape=(self.embedding_dim, self.num_classes),
            initializer='glorot_uniform',
            trainable=True
        )
        
    def call(self, inputs, labels=None, training=None):
        # Normalize embeddings and weights
        embeddings = tf.nn.l2_normalize(inputs, axis=1)
        weights = tf.nn.l2_normalize(self.W, axis=0)
        
        # Cosine similarity
        cos_theta = tf.matmul(embeddings, weights)
        cos_theta = tf.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        
        if training and labels is not None:
            # Add angular margin
            theta = tf.acos(cos_theta)
            one_hot = tf.one_hot(labels, self.num_classes)
            
            # Add margin to target class
            theta_m = theta + one_hot * self.margin
            cos_theta_m = tf.cos(theta_m)
            
            # Use margin-added cosine for target class
            logits = self.scale * cos_theta_m
        else:
            logits = self.scale * cos_theta
            
        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'embedding_dim': self.embedding_dim,
            'scale': self.scale,
            'margin': self.margin
        })
        return config


def load_dataset():
    """Load images and labels from raw directory."""
    print("Loading dataset...")
    
    images = []
    labels = []
    label_names = []
    
    part_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    
    for part_idx, part_dir in enumerate(part_dirs):
        part_id = part_dir.name
        label_names.append(part_id)
        
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


def create_embedding_model():
    """Create backbone model that outputs embeddings."""
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg"
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(EMBEDDING_DIM, name="embeddings")(x)
    # L2 normalize using UnitNormalization layer (serializable)
    embeddings = layers.UnitNormalization(axis=1)(x)
    
    model = Model(inputs, embeddings, name="embedding_model")
    return model


def create_arcface_model(embedding_model, num_classes):
    """Create full ArcFace training model."""
    
    inputs = embedding_model.input
    embeddings = embedding_model.output
    
    # Add ArcFace layer
    arcface = ArcFaceLayer(num_classes, EMBEDDING_DIM, ARCFACE_SCALE, ARCFACE_MARGIN)
    
    # Create training model with labels input
    labels_input = keras.Input(shape=(), dtype=tf.int32, name="labels")
    logits = arcface(embeddings, labels_input, training=True)
    
    model = Model([inputs, labels_input], logits, name="arcface_model")
    return model, arcface


class ArcFaceTrainer(keras.Model):
    """Custom training wrapper for ArcFace."""
    
    def __init__(self, embedding_model, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.embedding_model = embedding_model
        self.arcface = ArcFaceLayer(num_classes, EMBEDDING_DIM, ARCFACE_SCALE, ARCFACE_MARGIN)
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
            loss = tf.reduce_mean(loss)
        
        # Update weights
        trainable_vars = self.embedding_model.trainable_variables + self.arcface.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(labels, logits)
        
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_tracker.result()}
    
    def test_step(self, data):
        images, labels = data
        embeddings = self.embedding_model(images, training=False)
        logits = self.arcface(embeddings, labels, training=False)
        loss = keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        
        self.loss_tracker.update_state(tf.reduce_mean(loss))
        self.accuracy_tracker.update_state(labels, logits)
        
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_tracker.result()}
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]


def visualize_embeddings(embedding_model, image_paths, labels, label_names, output_path):
    """Visualize embeddings using t-SNE."""
    print("\nGenerating embedding visualization...")
    
    # Sample if too many images
    max_samples = 200
    if len(image_paths) > max_samples:
        indices = np.random.choice(len(image_paths), max_samples, replace=False)
        image_paths = [image_paths[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    # Get embeddings
    embeddings = []
    for path in image_paths:
        img = preprocess_image(path)
        img = tf.expand_dims(img, 0)
        emb = embedding_model(img, training=False)
        embeddings.append(emb.numpy()[0])
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=label_names[label] if label < len(label_names) else str(label),
            alpha=0.7
        )
    
    plt.title("LEGO Part Embeddings (t-SNE)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Embedding visualization saved: {output_path}")


def main():
    print("=" * 60)
    print("LEGO Part Recognition - ArcFace Metric Learning")
    print("=" * 60)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {len(gpus) > 0}")
    
    # Load data
    image_paths, labels, label_names = load_dataset()
    num_classes = len(label_names)
    
    # Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"\nDataset: {len(train_paths)} train, {len(val_paths)} val, {num_classes} classes")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"ArcFace scale: {ARCFACE_SCALE}, margin: {ARCFACE_MARGIN}")
    
    # Create datasets
    train_ds = create_dataset(train_paths, train_labels, training=True)
    val_ds = create_dataset(val_paths, val_labels, training=False)
    
    # Create models
    print("\nCreating embedding model...")
    embedding_model = create_embedding_model()
    embedding_model.summary()
    
    # Create trainer
    trainer = ArcFaceTrainer(embedding_model, num_classes)
    trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    
    # Train
    print(f"\nTraining for {EPOCHS} epochs with ArcFace loss...")
    history = trainer.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )
    
    # Results
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    
    final_loss = history.history['loss'][-1]
    final_acc = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"Final Training Loss: {final_loss:.4f}")
    print(f"Final Training Accuracy: {final_acc:.2%}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.2%}")
    
    # Save embedding model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "lego_embedding_model.keras"
    embedding_model.save(model_path)
    print(f"\nEmbedding model saved: {model_path}")
    
    # Visualize embeddings
    viz_path = MODEL_DIR / "embedding_visualization.png"
    visualize_embeddings(embedding_model, image_paths, labels, label_names, viz_path)
    
    # Save training plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Val')
    ax1.set_title('ArcFace Loss')
    ax1.legend()
    ax2.plot(history.history['accuracy'], label='Train')
    ax2.plot(history.history['val_accuracy'], label='Val')
    ax2.set_title('Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "arcface_training.png")
    print(f"Training plot saved: {MODEL_DIR / 'arcface_training.png'}")
    
    print("\n" + "=" * 60)
    print("[OK] ArcFace training complete!")
    print("The embedding model can now be used for similarity-based matching.")
    print("=" * 60)


if __name__ == "__main__":
    main()
