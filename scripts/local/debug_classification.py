"""
Debug Classification Script
Diagnoses why part 3001 (red 2x4 brick) is misclassified.

Usage:
    python scripts/local/debug_classification.py
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

# Setup paths
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(PROJECT_ROOT)

# Configuration
IMG_SIZE = (224, 224)
DB_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings", "hybrid_embeddings.pkl")
INPUT_IMAGE = os.path.join(PROJECT_ROOT, "data", "captures", "pi_camera_test.jpg")
DEBUG_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "debug_output")
TOP_N = 50  # Extended search

# Ensure output directory exists
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)


def load_database():
    """Load the embedding database."""
    print("=" * 80)
    print("STEP 1: DATABASE CHECK")
    print("=" * 80)

    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)

    print(f"Database loaded: {len(db)} entries")

    # Find all 3001 entries
    entries_3001 = []
    sources = {"legacy": 0, "real": 0, "b200c": 0}

    for key, entry in db.items():
        part_id = entry.get("part_id", key.split("_")[0])
        source = entry.get("source", "legacy")

        if part_id == "3001":
            entries_3001.append(
                {
                    "key": key,
                    "source": source,
                    "color_id": entry.get("color_id", "unknown"),
                }
            )
            sources[source] = sources.get(source, 0) + 1

    print(f"\nPart 3001 entries: {len(entries_3001)}")
    for src, count in sources.items():
        if count > 0:
            print(f"  {src}: {count}")

    if entries_3001:
        print(f"\nSample 3001 keys:")
        for e in entries_3001[:5]:
            print(f"  {e['key']} (color: {e['color_id']}, source: {e['source']})")

    # Find upn0389pr0005
    print(f"\nSearching for 'upn0389pr0005'...")
    upn_entries = []
    for key, entry in db.items():
        if (
            "upn0389pr0005" in key.lower()
            or entry.get("part_id", "") == "upn0389pr0005"
        ):
            upn_entries.append({"key": key, "entry": entry})

    if upn_entries:
        print(f"Found {len(upn_entries)} entries:")
        for ue in upn_entries[:3]:
            e = ue["entry"]
            print(f"  Key: {ue['key']}")
            print(f"    part_id: {e.get('part_id', 'N/A')}")
            print(f"    source: {e.get('source', 'N/A')}")
            print(f"    color_id: {e.get('color_id', 'N/A')}")
    else:
        print("  NOT FOUND in database keys or part_ids")
        # Search more broadly
        for key in list(db.keys())[:10]:
            print(f"  Sample key format: {key}")

    return db, entries_3001


def build_model():
    """Build EfficientNetB0 embedding model."""
    print("\nBuilding model...")
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


def visualize_background_removal():
    """Save intermediate background removal outputs."""
    print("\n" + "=" * 80)
    print("STEP 2: VISUALIZE BACKGROUND REMOVAL")
    print("=" * 80)

    if not os.path.exists(INPUT_IMAGE):
        print(f"ERROR: Input image not found: {INPUT_IMAGE}")
        return None

    # Load original
    with open(INPUT_IMAGE, "rb") as f:
        original_bytes = f.read()

    # Save original
    original = Image.open(io.BytesIO(original_bytes))
    original.save(os.path.join(DEBUG_OUTPUT_DIR, "01_original.jpg"))
    print(f"Saved: 01_original.jpg ({original.size})")

    # Apply rembg
    print("Applying rembg...")
    removed_bytes = rembg_remove(original_bytes)
    removed_rgba = Image.open(io.BytesIO(removed_bytes)).convert("RGBA")
    removed_rgba.save(os.path.join(DEBUG_OUTPUT_DIR, "02_rembg_transparent.png"))
    print(f"Saved: 02_rembg_transparent.png")

    # Composite on white (as inference_api.py does)
    bg = Image.new("RGBA", removed_rgba.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(bg, removed_rgba)
    final_rgb = composite.convert("RGB")
    final_rgb.save(os.path.join(DEBUG_OUTPUT_DIR, "03_final_white_bg.jpg"))
    print(f"Saved: 03_final_white_bg.jpg")

    print(f"\nDebug images saved to: {DEBUG_OUTPUT_DIR}")
    print("Please visually inspect these images!")

    return original_bytes, removed_bytes


def ground_truth_test(db, model):
    """Test matching with a known database entry."""
    print("\n" + "=" * 80)
    print("STEP 3: GROUND TRUTH TEST")
    print("=" * 80)

    db_keys = list(db.keys())
    db_vectors = np.array([v["embedding"] for v in db.values()])

    # Find a 3001 entry (prefer b200c for clean synthetic)
    test_key = None
    for key in db_keys:
        if "3001" in key:
            test_key = key
            if "b200c" in key:
                break  # Prefer b200c

    if not test_key:
        print("ERROR: No 3001 entry found in database!")
        return

    test_entry = db[test_key]
    test_embedding = test_entry["embedding"]

    print(f"Query: {test_key}")
    print(f"  Part ID: {test_entry.get('part_id')}")
    print(f"  Source: {test_entry.get('source')}")

    # Find matches
    sims = np.dot(db_vectors, test_embedding)
    top_idx = np.argsort(sims)[::-1][:10]

    print(f"\nTop 10 Matches:")
    for rank, idx in enumerate(top_idx):
        key = db_keys[idx]
        entry = db[key]
        part_id = entry.get("part_id", "unknown")
        marker = (
            " <-- SELF"
            if key == test_key
            else (" <-- 3001" if part_id == "3001" else "")
        )
        print(
            f"  {rank+1:2}. [{sims[idx]:.4f}] {part_id:15} ({entry.get('source', 'legacy'):8}){marker}"
        )


def extended_similarity_search(db, model, image_bytes_with_rembg):
    """Run extended top-50 search."""
    print("\n" + "=" * 80)
    print("STEP 4: EXTENDED SIMILARITY SEARCH (Top 50)")
    print("=" * 80)

    db_keys = list(db.keys())
    db_vectors = np.array([v["embedding"] for v in db.values()])

    # Process image (WITH rembg already applied)
    img = Image.open(io.BytesIO(image_bytes_with_rembg)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    query_embedding = model.predict(np.expand_dims(img_array, 0), verbose=0)[0]

    # Search
    sims = np.dot(db_vectors, query_embedding)
    top_idx = np.argsort(sims)[::-1][:TOP_N]

    print(f"Query: Pi capture WITH background removal")
    print(f"\nTop {TOP_N} Matches:")
    print("-" * 80)

    found_3001_rank = None
    for rank, idx in enumerate(top_idx):
        key = db_keys[idx]
        entry = db[key]
        part_id = entry.get("part_id", "unknown")
        source = entry.get("source", "legacy")

        marker = ""
        if part_id == "3001":
            marker = " <-- 3001 FOUND"
            if found_3001_rank is None:
                found_3001_rank = rank + 1

        # Show top 15 + any 3001
        if rank < 15 or part_id == "3001":
            print(
                f"  {rank+1:2}. [{sims[idx]:.4f}] Part: {part_id:15} Src: {source:8} Key: {key[:50]}{marker}"
            )

    if found_3001_rank:
        print(f"\n*** 3001 first appears at rank: {found_3001_rank} ***")
    else:
        print(f"\n*** 3001 NOT FOUND in top {TOP_N}! ***")

    return query_embedding


def ab_test_background_removal(db, model, original_bytes):
    """Compare results with and without background removal."""
    print("\n" + "=" * 80)
    print("STEP 5: A/B TEST (Background Removal On vs Off)")
    print("=" * 80)

    db_keys = list(db.keys())
    db_vectors = np.array([v["embedding"] for v in db.values()])

    def get_embedding(img_bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        return model.predict(np.expand_dims(img_array, 0), verbose=0)[0]

    def find_matches(embedding, top_k=10):
        sims = np.dot(db_vectors, embedding)
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_idx:
            key = db_keys[idx]
            entry = db[key]
            results.append(
                {
                    "rank": len(results) + 1,
                    "score": sims[idx],
                    "part_id": entry.get("part_id", "unknown"),
                    "source": entry.get("source", "legacy"),
                }
            )
        return results

    # Test A: WITH background removal
    print("\n[A] WITH Background Removal:")
    removed_bytes = rembg_remove(original_bytes)
    emb_with = get_embedding(removed_bytes)
    matches_with = find_matches(emb_with)

    found_3001_with = None
    for m in matches_with:
        marker = ""
        if m["part_id"] == "3001":
            marker = " <-- TARGET"
            if found_3001_with is None:
                found_3001_with = m["rank"]
        print(
            f"  {m['rank']:2}. [{m['score']:.4f}] {m['part_id']:15} ({m['source']}){marker}"
        )

    # Test B: WITHOUT background removal
    print("\n[B] WITHOUT Background Removal:")
    emb_without = get_embedding(original_bytes)
    matches_without = find_matches(emb_without)

    found_3001_without = None
    for m in matches_without:
        marker = ""
        if m["part_id"] == "3001":
            marker = " <-- TARGET"
            if found_3001_without is None:
                found_3001_without = m["rank"]
        print(
            f"  {m['rank']:2}. [{m['score']:.4f}] {m['part_id']:15} ({m['source']}){marker}"
        )

    # Compare
    cosine_sim = np.dot(emb_with, emb_without)
    print(f"\nEmbedding Similarity (A vs B): {cosine_sim:.4f}")

    print(
        f"\n3001 Rank WITH rembg: {found_3001_with if found_3001_with else 'Not in top 10'}"
    )
    print(
        f"3001 Rank WITHOUT rembg: {found_3001_without if found_3001_without else 'Not in top 10'}"
    )

    if found_3001_without and (
        not found_3001_with or found_3001_without < found_3001_with
    ):
        print("\n*** FINDING: Better results WITHOUT background removal! ***")
        print("    Recommendation: Tune rembg or use hybrid HSV removal")
    elif found_3001_with and (
        not found_3001_without or found_3001_with < found_3001_without
    ):
        print("\n*** FINDING: Better results WITH background removal ***")
        print("    Issue may be database coverage, not background removal")


def main():
    print("=" * 80)
    print("LEGO SORTER CLASSIFICATION DEBUG")
    print("=" * 80)
    print(f"Database: {DB_PATH}")
    print(f"Test Image: {INPUT_IMAGE}")
    print(f"Output Dir: {DEBUG_OUTPUT_DIR}")

    # Step 1: Load database and check 3001
    db, entries_3001 = load_database()

    # Build model (used by multiple steps)
    model = build_model()

    # Step 2: Visualize background removal
    result = visualize_background_removal()
    if result is None:
        print("Skipping remaining steps (no input image)")
        return
    original_bytes, removed_bytes = result

    # Step 3: Ground truth test
    ground_truth_test(db, model)

    # Step 4: Extended similarity search
    extended_similarity_search(db, model, removed_bytes)

    # Step 5: A/B test
    ab_test_background_removal(db, model, original_bytes)

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)
    print(f"\nCheck debug images in: {DEBUG_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
