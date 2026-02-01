import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import argparse
import sys
from scipy.spatial.distance import cosine

# Configuration (Must match build_full_vector_db.py)
IMG_SIZE = (224, 224)

def load_and_preprocess_image(path):
    try:
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not read image at {path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def build_model():
    print("Initializing Model...")
    # Base model
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling="avg"
    )
    
    # L2 Normalization wrapper
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    # Wrap TF op in Lambda layer to allow KerasTensor flow
    embeddings = keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x)
    model = keras.Model(inputs, embeddings)
    print("Model initialized.")
    return model

def load_database(db_path):
    print(f"Loading database from {db_path}...")
    try:
        with open(db_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Database loaded. {len(data)} entries.")
        return data
    except Exception as e:
        print(f"Failed to load database: {e}")
        return None

def find_closest_matches(model, db, img_path, top_k=5):
    # 1. Process Input
    img_tensor = load_and_preprocess_image(img_path)
    if img_tensor is None:
        return

    # Add batch dimension
    img_batch = np.expand_dims(img_tensor, axis=0)

    # 2. Get Embedding
    query_embedding = model.predict(img_batch, verbose=0)[0]

    # 3. Search
    print(f"Searching for matches for: {os.path.basename(img_path)}...")
    results = []
    
    # Simple linear scan (sufficient for 80k items for now, usually < 1s)
    # Cosine distance = 1 - cosine_similarity. We want smallest distance.
    for filename, emb in db.items():
        # Scipy cosine is "distance" (0=same, 1=orthogonal, 2=opposite)
        # So lower is better.
        dist = cosine(query_embedding, emb)
        results.append((filename, dist))

    # Sort by distance (ascending)
    results.sort(key=lambda x: x[1])

    # 4. Display
    print("\nTop 5 Matches:")
    print("-" * 50)
    print(f"{'Filename':<40} | {'Dist':<10} | {'Sim':<10}")
    print("-" * 50)
    for i in range(min(top_k, len(results))):
        fname, dist = results[i]
        sim = 1.0 - dist
        print(f"{fname[:37]+'...' if len(fname)>37 else fname:<40} | {dist:.4f}     | {sim:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Lego Part Inference Tester")
    parser.add_argument("--image", type=str, help="Path to query image")
    parser.add_argument("--db", type=str, default=r"data/embeddings/legacy_embeddings.pkl", help="Path to database pickle")
    args = parser.parse_args()

    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir)) # lego-sorter-v2
    
    # Resolve DB Path
    db_path = args.db
    if not os.path.isabs(db_path):
        db_path = os.path.join(project_root, db_path)

    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    # Load Resources
    db = load_database(db_path)
    if not db:
        return

    model = build_model()

    # Interactive Loop if no image provided
    if args.image:
        find_closest_matches(model, db, args.image)
    else:
        print("\n--- Interactive Mode ---")
        while True:
            path = input("\nEnter image path (or 'q' to quit): ").strip().strip('"')
            if path.lower() == 'q':
                break
            if not os.path.exists(path):
                print("File not found.")
                continue
            
            find_closest_matches(model, db, path)

if __name__ == "__main__":
    main()
