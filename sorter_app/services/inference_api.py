"""
FastAPI Inference Service for LEGO Part Recognition.

This module provides a persistent HTTP API for identifying LEGO parts
from images using the pre-built vector database.

Endpoints:
    - GET  /v1/health  : Health check
    - POST /v1/predict : Identify LEGO part from image
"""

import os
import io
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from scipy.spatial.distance import cosine
from PIL import Image
from rembg import remove as rembg_remove

# Configuration
IMG_SIZE = (224, 224)
TOP_K = 5
ENABLE_BACKGROUND_REMOVAL = True  # M3.5: Remove background before inference
BACKGROUND_COLOR = (255, 255, 255)  # White background

# Resolve paths relative to project root
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))  # lego-sorter-v2
# Use hybrid DB if available, fallback to legacy
HYBRID_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "embeddings", "hybrid_embeddings.pkl")
LEGACY_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "embeddings", "legacy_embeddings.pkl")
DB_PATH = HYBRID_DB_PATH if os.path.exists(HYBRID_DB_PATH) else LEGACY_DB_PATH


class ModelLoader:
    """
    Singleton class to load and hold the model and database in memory.
    Ensures the model is only loaded once at startup.
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
        
        # 1. Load Database
        if not os.path.exists(DB_PATH):
            raise RuntimeError(f"Database not found: {DB_PATH}")
        
        with open(DB_PATH, 'rb') as f:
            raw_db = pickle.load(f)
        
        # Detect format: new hybrid format has dict values with 'embedding' key
        sample_value = next(iter(raw_db.values()))
        if isinstance(sample_value, dict) and 'embedding' in sample_value:
            # Hybrid format
            self.db = raw_db
            self.is_hybrid = True
            print(f"[ModelLoader] Loaded {len(self.db)} embeddings (HYBRID format).")
        else:
            # Legacy format: convert to hybrid-like structure
            self.db = {}
            for filename, embedding in raw_db.items():
                base = os.path.splitext(filename)[0]
                parts = base.split('_')
                self.db[filename] = {
                    'embedding': embedding,
                    'source': 'legacy',
                    'part_id': parts[0] if len(parts) >= 1 else 'unknown',
                    'color_id': parts[1] if len(parts) >= 2 else '9999'
                }
            self.is_hybrid = False
            print(f"[ModelLoader] Loaded {len(self.db)} embeddings (LEGACY format, converted).")
        
        # 2. Build Model
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            pooling="avg"
        )
        inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        x = base_model(inputs, training=False)
        embeddings = keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x)
        self.model = keras.Model(inputs, embeddings)
        print("[ModelLoader] Model initialized.")
        
        # 3. Pre-compute vectors for fast search
        self.filenames = list(self.db.keys())
        self.vectors = np.array([v['embedding'] for v in self.db.values()])
        
        self._initialized = True
        print("[ModelLoader] Ready.")
    
    def remove_background(self, image_bytes: bytes) -> bytes:
        """
        Remove background from image using rembg (U-2-Net).
        Returns image bytes with white background.
        """
        # Remove background (returns RGBA with transparent background)
        output = rembg_remove(image_bytes)
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(output)).convert("RGBA")
        
        # Create white background
        background = Image.new("RGBA", img.size, BACKGROUND_COLOR + (255,))
        
        # Composite: paste image on white background
        composite = Image.alpha_composite(background, img)
        
        # Convert back to RGB bytes
        rgb_img = composite.convert("RGB")
        buffer = io.BytesIO()
        rgb_img.save(buffer, format="JPEG")
        return buffer.getvalue()
    
    def predict(self, image_bytes: bytes) -> list:
        """
        Predict the LEGO part from image bytes.
        Returns list of (filename, distance, similarity) tuples.
        """
        # 0. Background Removal (M3.5)
        if ENABLE_BACKGROUND_REMOVAL:
            try:
                image_bytes = self.remove_background(image_bytes)
            except Exception as e:
                print(f"[ModelLoader] Background removal failed: {e}, using original image")
        
        # 1. Preprocess Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_batch = np.expand_dims(img_array, axis=0)
        
        # 2. Get Embedding
        query_embedding = self.model.predict(img_batch, verbose=0)[0]
        
        # 3. Search (Dot product = Cosine Similarity for normalized vectors)
        similarities = np.dot(self.vectors, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:TOP_K]
        
        results = []
        for idx in top_indices:
            filename = self.filenames[idx]
            entry = self.db[filename]
            sim = float(similarities[idx])
            
            results.append({
                "part_id": entry.get('part_id', 'unknown'),
                "color_id": entry.get('color_id', '9999'),
                "source": entry.get('source', 'legacy'),
                "filename": filename,
                "confidence": round(sim, 4)
            })
        
        return results


# --- FastAPI App ---
app = FastAPI(
    title="LEGO Part Recognition API",
    version="1.0.0",
    description="API for identifying LEGO parts from images."
)

# Global model loader instance
loader = ModelLoader()


@app.on_event("startup")
async def startup_event():
    """Load model on startup to avoid cold start."""
    loader.initialize()


@app.get("/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": loader._initialized,
        "database_size": len(loader.db) if loader._initialized else 0
    }


@app.post("/v1/predict")
async def predict(image: UploadFile = File(...)):
    """
    Predict LEGO part from uploaded image.
    
    Args:
        image: Image file (JPEG/PNG)
    
    Returns:
        JSON with top 5 matches
    """
    if not loader._initialized:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    # Validate content type (relaxed to handle client variations)
    valid_types = ["image/jpeg", "image/png", "application/octet-stream", None]
    if image.content_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {image.content_type}. Use JPEG/PNG.")
    
    try:
        contents = await image.read()
        matches = loader.predict(contents)
        
        return {
            "success": True,
            "matches": matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
