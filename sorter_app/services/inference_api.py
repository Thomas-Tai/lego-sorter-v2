"""
FastAPI Inference Service for LEGO Part Recognition.

This module provides a persistent HTTP API for identifying LEGO parts
from images using either:
- M4 Classifier: ONNX model with direct classification (preferred)
- Legacy: Embedding similarity search (fallback)

Endpoints:
    - GET  /v1/health  : Health check
    - POST /v1/predict : Identify LEGO part from image
    - GET  /v1/info    : Model information
"""

import io
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from rembg import remove as rembg_remove

# Configuration
IMG_SIZE = (224, 224)
TOP_K = 5
# Background removal: Enable for production (Pi camera captures have real backgrounds)
# Disable for testing with pre-processed images (raw_clean/ already has white backgrounds)
# The classifier expects clean white-background images as that's what it was trained on.
ENABLE_BACKGROUND_REMOVAL = True
BACKGROUND_COLOR = (255, 255, 255)
# Test-Time Augmentation (TTA): Run multiple predictions with augmentations and average
# This improves accuracy by ~5-10% at the cost of 4x inference time
ENABLE_TTA = False

# Legacy penalties (only used in fallback mode)
LEGACY_PENALTY = 0.85
B200C_PENALTY = 0.95

# Resolve paths relative to project root
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent  # lego-sorter-v2

# M4 Classifier paths
CLASSIFIER_MODEL_PATH = _PROJECT_ROOT / "models" / "lego_classifier.onnx"
PART_MAPPING_PATH = _PROJECT_ROOT / "models" / "part_mapping.json"
COLOR_MAPPING_PATH = _PROJECT_ROOT / "models" / "color_mapping.json"

# Legacy embedding paths (fallback)
HYBRID_DB_PATH = _PROJECT_ROOT / "data" / "embeddings" / "hybrid_embeddings.pkl"
LEGACY_DB_PATH = _PROJECT_ROOT / "data" / "embeddings" / "legacy_embeddings.pkl"


class ClassifierInference:
    """
    M4 Classifier-based inference using ONNX Runtime.

    Direct classification of parts and colors without embedding search.
    """

    def __init__(
        self,
        model_path: Path,
        part_mapping_path: Path,
        color_mapping_path: Path,
    ):
        self.model_path = model_path
        self.session = None
        self.part_mapping: Dict[int, str] = {}
        self.color_mapping: Dict[int, str] = {}
        self._initialized = False

    def initialize(self):
        """Load ONNX model and class mappings."""
        if self._initialized:
            return

        import onnxruntime as ort

        print(f"[Classifier] Loading model: {self.model_path}")

        # Load ONNX model
        providers = ["CPUExecutionProvider"]
        try:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
                print("[Classifier] Using CUDA acceleration")
        except Exception:
            pass

        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # Load part mapping
        with open(PART_MAPPING_PATH, "r") as f:
            part_data = json.load(f)
            self.part_mapping = {
                int(k): v for k, v in part_data.get("idx_to_part", {}).items()
            }
        print(f"[Classifier] Loaded {len(self.part_mapping)} part classes")

        # Load color mapping
        with open(COLOR_MAPPING_PATH, "r") as f:
            color_data = json.load(f)
            self.color_mapping = {
                int(k): v for k, v in color_data.get("idx_to_color", {}).items()
            }
        print(f"[Classifier] Loaded {len(self.color_mapping)} color classes")

        self._initialized = True
        print("[Classifier] Ready")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for ONNX model (EfficientNet normalization)."""
        img = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)

        # EfficientNet preprocessing (same as tf.keras.applications.efficientnet)
        # Rescale to [0, 1] then normalize
        img_array = img_array / 255.0

        # ImageNet mean/std (approximation used by EfficientNet)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std

        # Add batch dimension and transpose to NCHW
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.transpose(img_array, (0, 3, 1, 2))

        return img_array.astype(np.float32)

    def _get_tta_images(self, img: Image.Image) -> List[Image.Image]:
        """Generate augmented versions of the image for TTA."""
        images = [img]  # Original
        images.append(img.transpose(Image.FLIP_LEFT_RIGHT))  # Horizontal flip
        images.append(img.rotate(90, expand=False, fillcolor=(255, 255, 255)))  # 90°
        images.append(img.rotate(270, expand=False, fillcolor=(255, 255, 255)))  # 270°
        return images

    def predict(self, image_bytes: bytes, top_k: int = TOP_K) -> List[Dict]:
        """
        Classify LEGO part from image bytes.

        Returns list of predictions with part_id, color_id, and confidence.
        Uses Test-Time Augmentation (TTA) if enabled for better accuracy.
        """
        # Load image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if ENABLE_TTA:
            # Run TTA: average predictions from multiple augmentations
            tta_images = self._get_tta_images(img)
            all_part_probs = []
            all_color_probs = []

            for aug_img in tta_images:
                input_tensor = self.preprocess(aug_img)
                outputs = self.session.run(None, {self.input_name: input_tensor})
                part_logits, color_logits = outputs
                all_part_probs.append(self._softmax(part_logits[0]))
                all_color_probs.append(self._softmax(color_logits[0]))

            # Average probabilities
            part_probs = np.mean(all_part_probs, axis=0)
            color_probs = np.mean(all_color_probs, axis=0)
        else:
            # Single inference
            input_tensor = self.preprocess(img)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            part_logits, color_logits = outputs
            part_probs = self._softmax(part_logits[0])
            color_probs = self._softmax(color_logits[0])

        # Get top-k part predictions
        top_part_indices = np.argsort(part_probs)[::-1][:top_k]
        top_color_idx = np.argmax(color_probs)

        results = []
        for rank, part_idx in enumerate(top_part_indices):
            part_id = self.part_mapping.get(part_idx, str(part_idx))
            color_id = self.color_mapping.get(top_color_idx, str(top_color_idx))

            results.append(
                {
                    "part_id": part_id,
                    "color_id": color_id,
                    "source": "classifier" + ("+tta" if ENABLE_TTA else ""),
                    "filename": f"{part_id}_{color_id}_pred",
                    "confidence": round(float(part_probs[part_idx]), 4),
                    "color_confidence": round(float(color_probs[top_color_idx]), 4),
                    "rank": rank + 1,
                }
            )

        return results

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)


class EmbeddingInference:
    """
    Legacy embedding-based inference using similarity search.

    Fallback when classifier model is not available.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db = None
        self.filenames = None
        self.vectors = None
        self.model = None
        self._initialized = False

    def initialize(self):
        """Load embedding database and TensorFlow model."""
        if self._initialized:
            return

        import tensorflow as tf
        from tensorflow import keras

        print(f"[Embedding] Loading database: {self.db_path}")

        # Load database
        with open(self.db_path, "rb") as f:
            raw_db = pickle.load(f)

        # Detect format
        sample_value = next(iter(raw_db.values()))
        if isinstance(sample_value, dict) and "embedding" in sample_value:
            self.db = raw_db
        else:
            # Convert legacy format
            self.db = {}
            for filename, embedding in raw_db.items():
                base = os.path.splitext(filename)[0]
                parts = base.split("_")
                self.db[filename] = {
                    "embedding": embedding,
                    "source": "legacy",
                    "part_id": parts[0] if len(parts) >= 1 else "unknown",
                    "color_id": parts[1] if len(parts) >= 2 else "9999",
                }

        print(f"[Embedding] Loaded {len(self.db)} embeddings")

        # Build model
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            pooling="avg",
        )
        inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        x = base_model(inputs, training=False)
        embeddings = keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x)
        self.model = keras.Model(inputs, embeddings)

        # Pre-compute vectors
        self.filenames = list(self.db.keys())
        self.vectors = np.array([v["embedding"] for v in self.db.values()])

        self._initialized = True
        print("[Embedding] Ready")

    def predict(self, image_bytes: bytes, top_k: int = TOP_K) -> List[Dict]:
        """Predict using embedding similarity search."""
        import tensorflow as tf

        # Preprocess
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_batch = np.expand_dims(img_array, axis=0)

        # Get embedding
        query_embedding = self.model.predict(img_batch, verbose=0)[0]

        # Search
        similarities = np.dot(self.vectors, query_embedding)

        # Apply source weighting
        adjusted = similarities.copy()
        for idx, filename in enumerate(self.filenames):
            source = self.db[filename].get("source", "legacy")
            if source == "legacy":
                adjusted[idx] *= LEGACY_PENALTY
            elif source == "b200c":
                adjusted[idx] *= B200C_PENALTY

        # Get top-k
        top_indices = np.argsort(adjusted)[::-1][:top_k]

        results = []
        for idx in top_indices:
            filename = self.filenames[idx]
            entry = self.db[filename]
            results.append(
                {
                    "part_id": entry.get("part_id", "unknown"),
                    "color_id": entry.get("color_id", "9999"),
                    "source": entry.get("source", "legacy"),
                    "filename": filename,
                    "confidence": round(float(similarities[idx]), 4),
                }
            )

        return results


class ModelLoader:
    """
    Unified model loader supporting both classifier and embedding inference.

    Automatically selects the best available inference method:
    1. M4 Classifier (ONNX) - preferred
    2. Embedding similarity - fallback
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized:
            return

        print("[ModelLoader] Initializing...")

        self.inference_mode = None
        self.classifier: Optional[ClassifierInference] = None
        self.embedding: Optional[EmbeddingInference] = None

        # Try M4 Classifier first
        if (
            CLASSIFIER_MODEL_PATH.exists()
            and PART_MAPPING_PATH.exists()
            and COLOR_MAPPING_PATH.exists()
        ):
            try:
                self.classifier = ClassifierInference(
                    CLASSIFIER_MODEL_PATH,
                    PART_MAPPING_PATH,
                    COLOR_MAPPING_PATH,
                )
                self.classifier.initialize()
                self.inference_mode = "classifier"
                print("[ModelLoader] Using M4 Classifier (ONNX)")
            except Exception as e:
                print(f"[ModelLoader] Classifier init failed: {e}")
                self.classifier = None

        # Fallback to embedding if classifier not available
        if self.inference_mode is None:
            db_path = HYBRID_DB_PATH if HYBRID_DB_PATH.exists() else LEGACY_DB_PATH
            if db_path.exists():
                try:
                    self.embedding = EmbeddingInference(db_path)
                    self.embedding.initialize()
                    self.inference_mode = "embedding"
                    print("[ModelLoader] Using Embedding Search (fallback)")
                except Exception as e:
                    print(f"[ModelLoader] Embedding init failed: {e}")

        if self.inference_mode is None:
            raise RuntimeError("No inference model available!")

        self._initialized = True
        print(f"[ModelLoader] Ready (mode: {self.inference_mode})")

    def remove_background(self, image_bytes: bytes) -> bytes:
        """Remove background using rembg."""
        output = rembg_remove(image_bytes)
        img = Image.open(io.BytesIO(output)).convert("RGBA")
        background = Image.new("RGBA", img.size, BACKGROUND_COLOR + (255,))
        composite = Image.alpha_composite(background, img)
        rgb_img = composite.convert("RGB")
        buffer = io.BytesIO()
        rgb_img.save(buffer, format="JPEG")
        return buffer.getvalue()

    def predict(self, image_bytes: bytes, top_k: int = TOP_K) -> List[Dict]:
        """Run inference using the selected method."""
        # Background removal
        if ENABLE_BACKGROUND_REMOVAL:
            try:
                image_bytes = self.remove_background(image_bytes)
            except Exception as e:
                print(f"[ModelLoader] Background removal failed: {e}")

        # Run inference
        if self.inference_mode == "classifier" and self.classifier:
            return self.classifier.predict(image_bytes, top_k)
        elif self.inference_mode == "embedding" and self.embedding:
            return self.embedding.predict(image_bytes, top_k)
        else:
            raise RuntimeError("No inference model initialized")

    def get_info(self) -> Dict:
        """Get model information."""
        info = {
            "inference_mode": self.inference_mode,
            "initialized": self._initialized,
        }

        if self.inference_mode == "classifier" and self.classifier:
            info["num_parts"] = len(self.classifier.part_mapping)
            info["num_colors"] = len(self.classifier.color_mapping)
            info["model_path"] = str(self.classifier.model_path)
        elif self.inference_mode == "embedding" and self.embedding:
            info["database_size"] = len(self.embedding.db)
            info["database_path"] = str(self.embedding.db_path)

        return info


# --- FastAPI App ---
app = FastAPI(
    title="LEGO Part Recognition API",
    version="2.0.0",
    description="API for identifying LEGO parts from images using M4 classifier.",
)

loader = ModelLoader()


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    loader.initialize()


@app.get("/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": loader._initialized,
        "inference_mode": loader.inference_mode,
    }


@app.get("/v1/info")
async def model_info():
    """Model information endpoint."""
    if not loader._initialized:
        raise HTTPException(status_code=503, detail="Model not ready")
    return loader.get_info()


@app.post("/v1/predict")
async def predict(image: UploadFile = File(...)):
    """
    Predict LEGO part from uploaded image.

    Args:
        image: Image file (JPEG/PNG)

    Returns:
        JSON with top matches including part_id, color_id, and confidence
    """
    if not loader._initialized:
        raise HTTPException(status_code=503, detail="Model not ready")

    valid_types = ["image/jpeg", "image/png", "application/octet-stream", None]
    if image.content_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {image.content_type}",
        )

    try:
        contents = await image.read()
        matches = loader.predict(contents)
        return {
            "success": True,
            "inference_mode": loader.inference_mode,
            "matches": matches,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
