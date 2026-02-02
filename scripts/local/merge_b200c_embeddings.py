"""
Merge B200C embeddings into the hybrid vector database.

This script:
1. Loads the existing Hybrid database (legacy + real captures).
2. Scans data/images/b200c_processed/ for processed B200C images.
3. Computes embeddings for B200C images.
4. Saves an updated hybrid database with all three sources.

Usage:
    python merge_b200c_embeddings.py
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
BATCH_SIZE = 64  # B200C images are already 224x224
B200C_PROCESSED_DIR = r"C:\D\WorkSpace\[Local]_Station\01_Heavy_Assets\LegoSorterProject\Data\images\b200c_processed"


def load_and_preprocess_image(path):
    """Load and preprocess an image for embedding."""
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # B200C processed images are already 224x224, but resize to be safe
        if img.shape[:2] != IMG_SIZE:
            img = cv2.resize(img, IMG_SIZE)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def build_model():
    """Build the embedding model (same as other scripts)."""
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), pooling="avg"
    )
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    embeddings = keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x)
    return keras.Model(inputs, embeddings)


def parse_b200c_filename(filename):
    """Parse part_id and view_index from B200C filename.

    Expected format: partid_vXXXX.jpg (e.g., 3001_v0080.jpg)
    """
    base = os.path.splitext(filename)[0]
    parts = base.rsplit("_v", 1)
    if len(parts) == 2:
        part_id = parts[0]
        view_idx = parts[1]
    else:
        part_id = base
        view_idx = "0000"
    return part_id, view_idx


def scan_b200c_processed(b200c_dir):
    """Scan b200c_processed/ for images and return list of (path, part_id, view_idx)."""
    images = []
    if not os.path.exists(b200c_dir):
        return images

    for filename in os.listdir(b200c_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(b200c_dir, filename)
            part_id, view_idx = parse_b200c_filename(filename)
            images.append((full_path, part_id, view_idx, filename))

    return images


def main():
    # GPU setup
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU Enabled: {len(gpus)} device(s) found.")
        except RuntimeError as e:
            print(f"GPU Error: {e}")

    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    hybrid_db_path = os.path.join(project_root, "data", "embeddings", "hybrid_embeddings.pkl")
    b200c_dir = B200C_PROCESSED_DIR

    # 1. Load existing Hybrid DB
    print(f"Loading Hybrid DB from {hybrid_db_path}...")
    if not os.path.exists(hybrid_db_path):
        print("Hybrid DB not found! Please run merge_real_captures.py first.")
        return

    with open(hybrid_db_path, "rb") as f:
        hybrid_db = pickle.load(f)

    # Count existing sources
    source_counts = {}
    for v in hybrid_db.values():
        src = v.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"Loaded {len(hybrid_db)} entries:")
    for src, count in sorted(source_counts.items()):
        print(f"  - {src}: {count}")

    # 2. Scan B200C processed images
    print(f"\nScanning B200C images in {b200c_dir}...")
    b200c_images = scan_b200c_processed(b200c_dir)
    print(f"Found {len(b200c_images)} B200C images.")

    if not b200c_images:
        print("No B200C images to process.")
        return

    # Check how many are already in DB
    existing_b200c = sum(1 for k in hybrid_db if k.startswith("b200c_"))
    print(f"Already in DB: {existing_b200c} B200C entries.")

    # Filter out already processed
    new_images = []
    for item in b200c_images:
        full_path, part_id, view_idx, filename = item
        key = f"b200c_{part_id}_v{view_idx}"
        if key not in hybrid_db:
            new_images.append(item)

    print(f"New images to embed: {len(new_images)}")

    if not new_images:
        print("All B200C images already in database.")
        return

    # 3. Build Model
    print("\nBuilding embedding model...")
    model = build_model()
    print("Model ready.")

    # 4. Process B200C images in batches
    print("\nProcessing B200C images...")
    start_time = time.time()
    processed = 0

    for i in range(0, len(new_images), BATCH_SIZE):
        batch = new_images[i : i + BATCH_SIZE]
        batch_images = []
        valid_entries = []

        for full_path, part_id, view_idx, filename in batch:
            img = load_and_preprocess_image(full_path)
            if img is not None:
                batch_images.append(img)
                valid_entries.append((part_id, view_idx))

        if not batch_images:
            continue

        # Compute embeddings
        batch_tensor = np.array(batch_images)
        embeddings = model.predict(batch_tensor, verbose=0)

        # Store in hybrid DB
        for idx, (part_id, view_idx) in enumerate(valid_entries):
            key = f"b200c_{part_id}_v{view_idx}"
            hybrid_db[key] = {
                "embedding": embeddings[idx],
                "source": "b200c",
                "part_id": part_id,
                "color_id": "9999",  # B200C doesn't have color info, use placeholder
            }
            processed += 1

        # Progress
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        sys.stdout.write(f"\rProcessed: {processed}/{len(new_images)} | Rate: {rate:.1f} img/s")
        sys.stdout.flush()

    print(f"\n\nDone! Processed {processed} B200C images.")

    # 5. Save updated Hybrid DB
    print(f"\nSaving updated Hybrid DB to {hybrid_db_path}...")
    with open(hybrid_db_path, "wb") as f:
        pickle.dump(hybrid_db, f)

    # Final stats
    source_counts = {}
    for v in hybrid_db.values():
        src = v.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\nUpdated Hybrid DB stats:")
    for src, count in sorted(source_counts.items()):
        print(f"  - {src}: {count}")
    print(f"  Total: {len(hybrid_db)}")


if __name__ == "__main__":
    main()
