#!/usr/bin/env python
"""
LEGO Part Similarity Inference
Identifies LEGO parts by comparing embeddings to reference database.
"""

import os
import sys
from pathlib import Path
import pickle
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Configuration
MODEL_DIR = PROJECT_ROOT / "models"
DB_DIR = PROJECT_ROOT / "data" / "embeddings"
IMG_SIZE = 224
TOP_K = 5  # Number of top matches to return


class LEGOPartMatcher:
    """Similarity-based LEGO part matching."""

    def __init__(self, model_path=None, db_path=None):
        """Initialize matcher with model and reference database."""
        self.model_path = model_path or MODEL_DIR / "lego_embedding_model.keras"
        self.db_path = db_path or DB_DIR / "reference_embeddings.pkl"

        self.model = None
        self.reference_db = None
        self.part_ids = []
        self.reference_embeddings = None

    def load(self):
        """Load model and reference database."""
        print("Loading embedding model...")
        self.model = keras.models.load_model(self.model_path, safe_mode=False)

        print("Loading reference database...")
        with open(self.db_path, "rb") as f:
            self.reference_db = pickle.load(f)

        # Pre-compute reference embedding matrix for fast similarity
        self.part_ids = sorted(self.reference_db.keys())
        self.reference_embeddings = np.array([self.reference_db[pid]["embedding"] for pid in self.part_ids])

        print(f"Loaded {len(self.part_ids)} parts")
        return self

    def preprocess_image(self, image_path):
        """Load and preprocess image."""
        img = tf.io.read_file(str(image_path))
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img

    def get_embedding(self, image_path):
        """Get embedding for an image."""
        img = self.preprocess_image(image_path)
        img = tf.expand_dims(img, 0)
        embedding = self.model(img, training=False)
        return embedding.numpy()[0]

    def match(self, image_path, top_k=TOP_K):
        """Find most similar parts for an image.

        Args:
            image_path: Path to query image
            top_k: Number of top matches to return

        Returns:
            List of (part_id, similarity_score) tuples
        """
        # Get query embedding
        query_embedding = self.get_embedding(image_path)

        # Compute cosine similarity with all references
        # Since embeddings are L2 normalized, dot product = cosine similarity
        similarities = np.dot(self.reference_embeddings, query_embedding)

        # Get top-k matches
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            part_id = self.part_ids[idx]
            similarity = similarities[idx]
            results.append((part_id, float(similarity)))

        return results

    def match_batch(self, image_paths, top_k=TOP_K):
        """Match multiple images."""
        results = []
        for path in image_paths:
            matches = self.match(path, top_k)
            results.append({"image": str(path), "matches": matches})
        return results


def interactive_demo(matcher):
    """Interactive demo for testing."""
    print("\n" + "=" * 60)
    print("Interactive LEGO Part Matcher")
    print("Enter image path to identify, 'q' to quit")
    print("=" * 60)

    while True:
        path = input("\nImage path: ").strip()

        if path.lower() == "q":
            break

        if not Path(path).exists():
            print(f"File not found: {path}")
            continue

        start = time.time()
        matches = matcher.match(path)
        elapsed = time.time() - start

        print(f"\nTop {len(matches)} matches ({elapsed*1000:.1f}ms):")
        for i, (part_id, score) in enumerate(matches, 1):
            confidence = "HIGH" if score > 0.8 else "MEDIUM" if score > 0.6 else "LOW"
            print(f"  {i}. {part_id}: {score:.3f} ({confidence})")


def test_with_sample():
    """Test with a random sample image."""
    raw_dir = PROJECT_ROOT / "data" / "images" / "raw"

    # Get a sample image
    for part_dir in raw_dir.iterdir():
        if part_dir.is_dir():
            for color_dir in part_dir.iterdir():
                if color_dir.is_dir():
                    for img in color_dir.glob("*.jpg"):
                        return img
    return None


def main():
    print("=" * 60)
    print("LEGO Part Similarity Matcher")
    print("=" * 60)

    # Initialize matcher
    matcher = LEGOPartMatcher()
    matcher.load()

    # Test with sample
    sample = test_with_sample()
    if sample:
        print(f"\nTesting with sample: {sample.name}")
        expected_part = sample.parent.parent.name

        start = time.time()
        matches = matcher.match(sample)
        elapsed = time.time() - start

        print(f"\nExpected: {expected_part}")
        print(f"Inference time: {elapsed*1000:.1f}ms")
        print(f"\nTop {len(matches)} matches:")
        for i, (part_id, score) in enumerate(matches, 1):
            marker = " <-- CORRECT" if part_id == expected_part else ""
            print(f"  {i}. {part_id}: {score:.3f}{marker}")

        # Check if correct
        if matches[0][0] == expected_part:
            print("\n[OK] Correct match!")
        else:
            print("\n[WARN] Top match is incorrect")

    # Run interactive demo
    interactive_demo(matcher)


if __name__ == "__main__":
    main()
