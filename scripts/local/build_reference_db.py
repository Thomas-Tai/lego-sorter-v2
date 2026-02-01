#!/usr/bin/env python
"""
Build Reference Embedding Database
Creates a database of average embeddings for each LEGO part.
Used for similarity-based part identification.
"""
import os
import sys
from pathlib import Path
import json
import pickle

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import defaultdict

# Configuration
RAW_DIR = PROJECT_ROOT / "data" / "images" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
DB_DIR = PROJECT_ROOT / "data" / "embeddings"
IMG_SIZE = 224


def load_embedding_model():
    """Load trained embedding model."""
    model_path = MODEL_DIR / "lego_embedding_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Embedding model not found: {model_path}")
    
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path, safe_mode=False)
    return model


def preprocess_image(image_path):
    """Load and preprocess a single image."""
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def get_embedding(model, image_path):
    """Get embedding for a single image."""
    img = preprocess_image(image_path)
    img = tf.expand_dims(img, 0)
    embedding = model(img, training=False)
    return embedding.numpy()[0]


def build_reference_database(model):
    """Build reference embeddings for all parts."""
    print("\nBuilding reference database...")
    
    reference_db = {}
    part_embeddings = defaultdict(list)
    
    # Scan all parts
    part_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    total_images = 0
    
    for part_dir in part_dirs:
        part_id = part_dir.name
        
        # Get all images for this part
        for color_dir in part_dir.iterdir():
            if not color_dir.is_dir():
                continue
            for img_path in color_dir.glob("*.jpg"):
                embedding = get_embedding(model, img_path)
                part_embeddings[part_id].append(embedding)
                total_images += 1
        
        if part_id in part_embeddings:
            print(f"  {part_id}: {len(part_embeddings[part_id])} images")
    
    # Compute average embedding for each part
    for part_id, embeddings in part_embeddings.items():
        avg_embedding = np.mean(embeddings, axis=0)
        # L2 normalize
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        reference_db[part_id] = {
            'embedding': avg_embedding.tolist(),
            'num_samples': len(embeddings)
        }
    
    print(f"\nTotal: {len(reference_db)} parts, {total_images} images")
    return reference_db


def save_database(reference_db, output_dir):
    """Save reference database to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON (human readable)
    json_path = output_dir / "reference_embeddings.json"
    with open(json_path, 'w') as f:
        json.dump(reference_db, f, indent=2)
    print(f"Saved JSON: {json_path}")
    
    # Save as pickle (faster loading)
    pkl_path = output_dir / "reference_embeddings.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(reference_db, f)
    print(f"Saved pickle: {pkl_path}")
    
    # Save part list
    parts_path = output_dir / "part_list.txt"
    with open(parts_path, 'w') as f:
        for part_id in sorted(reference_db.keys()):
            f.write(f"{part_id}\n")
    print(f"Saved part list: {parts_path}")
    
    return json_path, pkl_path


def main():
    print("=" * 60)
    print("LEGO Part Reference Database Builder")
    print("=" * 60)
    
    # Load model
    model = load_embedding_model()
    
    # Build database
    reference_db = build_reference_database(model)
    
    # Save
    json_path, pkl_path = save_database(reference_db, DB_DIR)
    
    print("\n" + "=" * 60)
    print("[OK] Reference database built successfully!")
    print(f"Parts: {len(reference_db)}")
    print(f"Database: {pkl_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
