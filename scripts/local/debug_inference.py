"""
Debug Inference Script
Usage: python scripts/local/debug_inference.py
"""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from rembg import remove as rembg_remove
import io
import time
import random

# --- Configuration ---
IMG_SIZE = (224, 224)
TOP_K = 5
HEAVY_ASSETS_ROOT = r"C:\D\WorkSpace\[Local]_Station\01_Heavy_Assets\LegoSorterProject"
TEST_IMAGES_DIR = os.path.join(HEAVY_ASSETS_ROOT, "Data", "images", "raw")
PROJECT_ROOT = r"C:\D\WorkSpace\[Cloud]_Company_Sync\MSC\OwnInfo\MyResearchProject\Lego_Sorter_V2\CodeBase\lego-sorter-v2"
HYBRID_DB_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings", "hybrid_embeddings.pkl")

# --- Model Loader Logic (Simplified from inference_api.py) ---
def load_db():
    print(f"Loading DB from {HYBRID_DB_PATH}...")
    if not os.path.exists(HYBRID_DB_PATH):
        print("DB not found!")
        return None, None
    
    with open(HYBRID_DB_PATH, "rb") as f:
        db = pickle.load(f)
    print(f"Loaded {len(db)} entries.")
    
    # Check stats
    sources = {}
    for v in db.values():
        val = v if isinstance(v, dict) else {"source": "legacy"}
        src = val.get("source", "legacy")
        sources[src] = sources.get(src, 0) + 1
    
    print("DB Stats:", sources)
    return db

def build_model():
    print("Building EfficientNetB0 model...")
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling="avg",
    )
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    embeddings = keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x)
    model = keras.Model(inputs, embeddings)
    return model

def preprocess(image_path, model):
    try:
        # Load
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        
        # Rembg
        img_bytes = rembg_remove(img_bytes)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Resize & Preprocess
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Embed
        embedding = model.predict(img_batch, verbose=0)[0]
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def find_matches(query_vec, db_vectors, db_keys, top_k=5):
    # Dot product
    sims = np.dot(db_vectors, query_vec)
    top_indices = np.argsort(sims)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        key = db_keys[idx]
        score = sims[idx]
        results.append((key, score))
    return results

def main():
    # 1. Setup
    db = load_db()
    if not db: return
    
    # Prepare vectors
    db_keys = list(db.keys())
    db_vectors = np.array([
        v["embedding"] if isinstance(v, dict) else v 
        for v in db.values()
    ])
    
    model = build_model()
    
    # 2. Select Test Images
    # Look for part 3004 (1x2 Brick) and 3001 (2x4 Brick) if available
    test_parts = ["3004", "3020", "3001", "3024"] 
    
    print("\n--- Starting Test ---")
    
    for part_id in test_parts:
        part_dir = os.path.join(TEST_IMAGES_DIR, part_id)
        if not os.path.exists(part_dir):
            print(f"Skipping {part_id} (not found in raw)")
            continue
            
        # Recursive search for images
        images = []
        for root, dirs, files in os.walk(part_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images.append(os.path.join(root, file))
        
        if not images:
            print(f"Skipping {part_id} (no images found)")
            continue
            
        # Pick 2 random images per part
        samples = random.sample(images, min(2, len(images)))
        
        for fpath in samples:
            fname = os.path.basename(fpath)
            print(f"\nQuery: {part_id} | File: {fname}")
            
            t0 = time.time()
            vec = preprocess(fpath, model)
            if vec is None: continue
            
            matches = find_matches(vec, db_vectors, db_keys)
            dt = time.time() - t0
            
            print(f"Inference Time: {dt:.2f}s")
            print("Top 5 Matches:")
            found_correct = False
            for rank, (key, score) in enumerate(matches):
                # Parse DB entry
                entry = db[key]
                matched_part = entry.get("part_id", "unknown") if isinstance(entry, dict) else key.split("_")[0]
                source = entry.get("source", "legacy") if isinstance(entry, dict) else "legacy"
                
                is_correct = (matched_part == part_id)
                if is_correct: found_correct = True
                
                mark = "✅" if is_correct else "❌"
                print(f"  {rank+1}. {mark} [{score:.4f}] Part: {matched_part} (Src: {source}) - {key}")
            
            if not found_correct:
                print("  ⚠️ FAILED to find correct part in Top 5")

if __name__ == "__main__":
    main()
