"""
Merge Real Capture embeddings into the vector database.

This script:
1. Loads the existing Legacy database.
2. Scans data/images/raw/ for Real Capture images.
3. Computes embeddings for Real Captures.
4. Saves a hybrid database with both sources.
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import time

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def load_and_preprocess_image(path):
    """Load and preprocess an image for embedding."""
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def build_model():
    """Build the embedding model (same as build_full_vector_db.py)."""
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling="avg",
    )
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    embeddings = keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x)
    return keras.Model(inputs, embeddings)


def scan_real_captures(raw_dir):
    """Scan data/images/raw/ for images and return list of (path, part_id, color_id)."""
    images = []
    for part_id in os.listdir(raw_dir):
        part_path = os.path.join(raw_dir, part_id)
        if not os.path.isdir(part_path):
            continue
        for color_id in os.listdir(part_path):
            color_path = os.path.join(part_path, color_id)
            if not os.path.isdir(color_path):
                continue
            for filename in os.listdir(color_path):
                # Only use PNGs (which are the clean, background-removed versions)
                if filename.lower().endswith(".png"):
                    full_path = os.path.join(color_path, filename)
                    images.append((full_path, part_id, color_id, filename))
    return images


def main():
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    legacy_db_path = os.path.join(
        project_root, "data", "embeddings", "legacy_embeddings.pkl"
    )
    hybrid_db_path = os.path.join(
        project_root, "data", "embeddings", "hybrid_embeddings.pkl"
    )
    # Use background-removed images from raw_clean/ (M3.5)
    # Use background-removed images from Heavy Assets
    # raw_dir = os.path.join(project_root, "data", "images", "raw_clean")
    raw_dir = r"C:\D\WorkSpace\[Local]_Station\01_Heavy_Assets\LegoSorterProject\Data\images\raw_clean"

    # 1. Load Legacy DB
    print(f"Loading Legacy DB from {legacy_db_path}...")
    if not os.path.exists(legacy_db_path):
        print("Legacy DB not found!")
        return

    with open(legacy_db_path, "rb") as f:
        legacy_db = pickle.load(f)
    print(f"Loaded {len(legacy_db)} legacy entries.")

    # 2. Initialize hybrid DB with legacy data (add source metadata)
    # Format: {filename: {"embedding": np.array, "source": "legacy"/"real", "part_id": str, "color_id": str}}
    hybrid_db = {}
    for filename, embedding in legacy_db.items():
        # Parse part_id and color_id from filename (e.g., "3001_15.jpg")
        base = os.path.splitext(filename)[0]
        parts = base.split("_")
        part_id = parts[0] if len(parts) >= 1 else "unknown"
        color_id = parts[1] if len(parts) >= 2 else "9999"

        hybrid_db[filename] = {
            "embedding": embedding,
            "source": "legacy",
            "part_id": part_id,
            "color_id": color_id,
        }

    # 3. Scan Real Captures
    print(f"Scanning Real Captures in {raw_dir}...")
    if not os.path.exists(raw_dir):
        print("Raw directory not found!")
        return

    real_images = scan_real_captures(raw_dir)
    print(f"Found {len(real_images)} real capture images.")

    if not real_images:
        print("No real images to process. Saving hybrid DB as-is.")
        with open(hybrid_db_path, "wb") as f:
            pickle.dump(hybrid_db, f)
        return

    # 4. Build Model
    print("Building embedding model...")
    model = build_model()

    # 5. Process Real Captures in batches
    print("Processing Real Captures...")
    start_time = time.time()
    processed = 0

    for i in range(0, len(real_images), BATCH_SIZE):
        batch = real_images[i : i + BATCH_SIZE]
        batch_images = []
        valid_entries = []

        for full_path, part_id, color_id, filename in batch:
            img = load_and_preprocess_image(full_path)
            if img is not None:
                batch_images.append(img)
                valid_entries.append((filename, part_id, color_id))

        if not batch_images:
            continue

        # Compute embeddings
        batch_tensor = np.array(batch_images)
        embeddings = model.predict(batch_tensor, verbose=0)

        # Store in hybrid DB with "real_" prefix to avoid key collision
        for idx, (filename, part_id, color_id) in enumerate(valid_entries):
            key = f"real_{part_id}_{color_id}_{filename}"
            hybrid_db[key] = {
                "embedding": embeddings[idx],
                "source": "real",
                "part_id": part_id,
                "color_id": color_id,
            }
            processed += 1

        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        sys.stdout.write(
            f"\rProcessed: {processed}/{len(real_images)} | Rate: {rate:.1f} img/s"
        )
        sys.stdout.flush()

    print(f"\n\nDone! Processed {processed} real images.")

    # 6. Save Hybrid DB
    print(f"Saving Hybrid DB to {hybrid_db_path}...")
    with open(hybrid_db_path, "wb") as f:
        pickle.dump(hybrid_db, f)

    legacy_count = sum(1 for v in hybrid_db.values() if v["source"] == "legacy")
    real_count = sum(1 for v in hybrid_db.values() if v["source"] == "real")
    print(
        f"Hybrid DB saved: {legacy_count} legacy + {real_count} real = {len(hybrid_db)} total."
    )


if __name__ == "__main__":
    main()
