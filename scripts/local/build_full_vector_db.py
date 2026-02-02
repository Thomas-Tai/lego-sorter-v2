import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import time
import sys
from concurrent.futures import ThreadPoolExecutor

# Configuration
IMAGE_DIR = (
    r"C:\D\WorkSpace\[Local]_Station\01_Heavy_Assets\lego_inventory_parts_ID_Color"
)
BATCH_SIZE = 64
IMG_SIZE = (224, 224)


def load_and_preprocess_image(path):
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img
    except Exception as e:
        return None


def main():
    # 1. Global Setup
    # Enable GPU Memory Growth to prevent allocation errors
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU Enabled: {len(gpus)} device(s) found.")
        except RuntimeError as e:
            print(f"GPU Error: {e}")

    # Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))  # lego-sorter-v2

    # Use persistent data directory
    OUTPUT_DIR = os.path.join(project_root, "data", "embeddings")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "legacy_embeddings.pkl")

    # Fallback to TEMP if data dir is not writable
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
        except OSError:
            print(f"Warning: Could not create {OUTPUT_DIR}, falling back to TEMP")
            OUTPUT_DIR = os.environ.get("TEMP", ".")
            OUTPUT_FILE = os.path.join(OUTPUT_DIR, "legacy_embeddings.pkl")

    print(f"Output File: {OUTPUT_FILE}")

    # 2. Resume Logic
    embeddings_db = {}
    if os.path.exists(OUTPUT_FILE):
        print("Found existing database. Attempting to resume...")
        try:
            with open(OUTPUT_FILE, "rb") as f:
                embeddings_db = pickle.load(f)
            print(f"Resumed! Loaded {len(embeddings_db)} existing entries.")
        except Exception as e:
            print(f"Could not load existing DB ({e}). Starting fresh.")
            embeddings_db = {}

    # 3. Model Initialization
    print(f"Initializing EfficientNetB0 (ImageNet)...")
    try:
        # Use standard ImageNet backbone for robustness
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            pooling="avg",
        )
        # Add L2 normalization
        inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        x = base_model(inputs, training=False)
        embeddings = tf.nn.l2_normalize(x, axis=1)
        model = keras.Model(inputs, embeddings)
        print("Model initialized successfully (ImageNet Backbone).")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # 4. File Scanning
    print(f"Scanning images in {IMAGE_DIR}...")
    if not os.path.exists(IMAGE_DIR):
        print(f"Image directory not found: {IMAGE_DIR}")
        return

    all_files = [
        f
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    total_files = len(all_files)

    # Filter processed
    processed_files = set(embeddings_db.keys())
    files_to_process = [f for f in all_files if f not in processed_files]

    print(f"Total images: {total_files}")
    print(f"Already processed: {len(processed_files)}")
    print(f"Remaining to process: {len(files_to_process)}")

    if not files_to_process:
        print("All files processed!")
        return

    # 5. Processing Loop
    start_time = time.time()
    SAVE_EVERY = 1000
    last_save_count = 0

    # Process in batches
    for i in range(0, len(files_to_process), BATCH_SIZE):
        batch_paths = files_to_process[i : i + BATCH_SIZE]
        batch_images = []
        valid_paths = []

        # Load images
        for filename in batch_paths:
            full_path = os.path.join(IMAGE_DIR, filename)
            img = load_and_preprocess_image(full_path)
            if img is not None:
                batch_images.append(img)
                valid_paths.append(filename)

        if not batch_images:
            continue

        # Stack and Predict
        batch_tensor = np.array(batch_images)

        # Predict
        batch_embeddings = model.predict(batch_tensor, verbose=0)

        # Store
        for idx, filename in enumerate(valid_paths):
            embeddings_db[filename] = batch_embeddings[idx]

        # Logging & Auto-save
        current_count = len(embeddings_db) - len(processed_files)
        total_to_do = len(files_to_process)
        elapsed = time.time() - start_time
        rate = current_count / elapsed if elapsed > 0 else 0

        sys.stdout.write(
            f"\rProcessed: {current_count}/{total_to_do} ({(current_count/total_to_do)*100:.1f}%) | Rate: {rate:.1f} img/s | Total DB: {len(embeddings_db)}"
        )
        sys.stdout.flush()

        if current_count - last_save_count >= SAVE_EVERY:
            with open(OUTPUT_FILE, "wb") as f:
                pickle.dump(embeddings_db, f)
            last_save_count = current_count

    print("\n\nEncoding complete.")

    # 6. Final Save
    print(f"Saving database to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(embeddings_db, f)

    print(f"Database saved. Size: {len(embeddings_db)} entries.")


if __name__ == "__main__":
    main()
