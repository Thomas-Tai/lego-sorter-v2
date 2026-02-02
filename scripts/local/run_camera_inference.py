"""
Real-time Camera Inference Loop
Captures images from the webcam and queries the Lego Vector DB.

Usage:
    python scripts/local/run_camera_inference.py
"""

import os
import sys
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from rembg import remove as rembg_remove
from PIL import Image
import io

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from modules.hardware.camera import CameraDriver

# --- Configuration ---
IMG_SIZE = (224, 224)
TOP_K = 5
HYBRID_DB_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings", "hybrid_embeddings.pkl")
TEMP_CAPTURE_PATH = os.path.join(PROJECT_ROOT, "data", "captures", "live_test.jpg")

# --- Logic ---

def load_db():
    print(f"Loading DB from {HYBRID_DB_PATH}...")
    if not os.path.exists(HYBRID_DB_PATH):
        print("DB not found!")
        return None, None, None
    
    with open(HYBRID_DB_PATH, "rb") as f:
        db = pickle.load(f)
    print(f"Loaded {len(db)} entries.")
    
    db_keys = list(db.keys())
    db_vectors = np.array([
        v["embedding"] if isinstance(v, dict) else v 
        for v in db.values()
    ])
    return db, db_keys, db_vectors

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
        if not os.path.exists(image_path):
            return None

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
        print(f"Error processing image: {e}")
        return None

def find_matches(query_vec, db_vectors, db_keys, top_k=5):
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
    db, db_keys, db_vectors = load_db()
    if not db: return
    
    model = build_model()
    camera = CameraDriver(camera_index=0)
    
    print("\n--- Starting Camera Inference ---")
    if not camera.open():
        print("Failed to open camera! Check connection.")
        return

    try:
        while True:
            input("\nPress ENTER to capture and identify... (Ctrl+C to quit)")
            
            print("Capturing...")
            success = camera.capture(TEMP_CAPTURE_PATH)
            
            if not success:
                print("Capture failed.")
                continue
                
            print(f"Captured: {TEMP_CAPTURE_PATH}")
            
            t0 = time.time()
            vec = preprocess(TEMP_CAPTURE_PATH, model)
            if vec is None: continue
            
            matches = find_matches(vec, db_vectors, db_keys)
            dt = time.time() - t0
            
            print(f"Inference Time: {dt:.2f}s")
            
            print(f"Inference Time: {dt:.2f}s", flush=True)
            
            # Save results to file for reliable reading
            log_path = os.path.join(PROJECT_ROOT, "data", "captures", "inference.log")
            
            # Ensure directory exists first
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            
            try:
                with open(log_path, "w") as log:
                    log.write(f"Inference Time: {dt:.2f}s\n")
                    log.write("-" * 30 + "\n")
                    log.write("TOP MATCHES:\n")
                    print("-" * 30, flush=True)
                    print("TOP MATCHES:", flush=True)
                    for rank, (key, score) in enumerate(matches):
                        entry = db[key]
                        matched_part = entry.get("part_id", "unknown") if isinstance(entry, dict) else key.split("_")[0]
                        source = entry.get("source", "legacy") if isinstance(entry, dict) else "legacy"
                        line = f"  {rank+1}. [{score:.4f}] Part: {matched_part} (Src: {source})"
                        print(line, flush=True)
                        log.write(line + "\n")
                    print("-" * 30, flush=True)
                    log.write("-" * 30 + "\n")
            except Exception as e:
                print(f"Failed to write log: {e}", flush=True)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        camera.close()

if __name__ == "__main__":
    main()
