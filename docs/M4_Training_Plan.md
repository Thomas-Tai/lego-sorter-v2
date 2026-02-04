# M4: Model Training Plan

## Objective
Train a LEGO part classifier that accurately identifies **part ID** and **color ID** from real camera captures, using all available data sources (Legacy, B200C, Real).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Three-Stage Training Pipeline                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Stage 1: Synthetic Pre-training     Stage 2: Real Fine-tuning          │
│  ┌─────────────────────────────┐     ┌─────────────────────────────┐    │
│  │ Legacy (82k) + B200C (40k)  │     │ Real Captures (1.7k)        │    │
│  │           ↓                 │     │           ↓                 │    │
│  │ EfficientNetB0 (ImageNet)   │ ──> │ backbone_synthetic.pth      │    │
│  │           ↓                 │     │           ↓                 │    │
│  │ Learn LEGO shape features   │     │ Adapt to real camera domain │    │
│  │           ↓                 │     │           ↓                 │    │
│  │ backbone_synthetic.pth      │     │ backbone_final.pth          │    │
│  └─────────────────────────────┘     └─────────────────────────────┘    │
│                                                   │                      │
│                                                   ▼                      │
│                              ┌─────────────────────────────────────┐    │
│                              │ Stage 3: Deployment Options         │    │
│                              ├─────────────────────────────────────┤    │
│                              │                                     │    │
│                              │  Option A: Classifier Head (Demo)   │    │
│                              │  - 105 classes for set 45345-1      │    │
│                              │  - Accuracy: >95%                   │    │
│                              │                                     │    │
│                              │  Option B: Vector Space (Expansion) │    │
│                              │  - Re-extract embeddings            │    │
│                              │  - Add new sets without retraining  │    │
│                              │  - Accuracy: ~90%                   │    │
│                              │                                     │    │
│                              └─────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

### Available Data

| Source | Images | Classes | Size | Color Info | Domain |
|--------|--------|---------|------|------------|--------|
| **Legacy** | 82,354 | ~6,000 | 224×224 | ✅ Yes | Rendered |
| **B200C** | 800,000 | 200 | 64×64 | ❌ No | Synthetic 3D |
| **Real** | 1,687 | ~50 | 640×480 | ✅ Yes | Camera |

### B200C 64×64 Handling

```python
# B200C images are 64×64, need upscaling to 224×224

# Recommended: Lanczos interpolation + blur augmentation
transform_b200c = A.Compose([
    A.Resize(224, 224, interpolation=cv2.INTER_LANCZOS4),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.MotionBlur(blur_limit=(3, 7)),
    ], p=0.3),  # Match blur characteristics of upscaled images
])

# Alternative: Real-ESRGAN for quality upscaling (slower, better quality)
```

### Data Sampling Strategy

```python
training_data = {
    "legacy": 82000,      # All legacy images
    "b200c": 40000,       # Sample 200 views per part (200 × 200 = 40k)
    "real": 1700,         # All real captures (weighted 3x)
}

# Total: ~124k images for Stage 1
# Stage 2: Real only with heavy augmentation
```

---

## Stage 1: Synthetic Pre-training

### Objective
Learn LEGO part shape and structure features from synthetic data.

### Configuration

```yaml
# config/stage1_synthetic.yaml
model:
  backbone: efficientnet_b0
  pretrained: imagenet
  num_classes: 200  # B200C has 200 classes

data:
  sources:
    - legacy: 82000
    - b200c: 40000  # Sampled
  batch_size: 64
  image_size: 224

training:
  epochs: 15
  optimizer: AdamW
  learning_rate: 1e-4
  weight_decay: 0.01
  scheduler: CosineAnnealingLR

augmentation:
  - RandomRotate90
  - HorizontalFlip
  - RandomBrightnessContrast
  - GaussianBlur  # Important for B200C upscaled images
  - CoarseDropout
```

### Code

```python
# scripts/training/stage1_synthetic.py

import torch
import torch.nn as nn
from torchvision import models
import albumentations as A

class Stage1Model(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_features(self, x):
        """Extract features before classifier"""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        return x.flatten(1)

def train_stage1():
    model = Stage1Model(num_classes=200)

    # Freeze early layers initially
    for param in model.backbone.features[:5].parameters():
        param.requires_grad = False

    # Training loop...
    # Save: checkpoints/backbone_synthetic.pth
```

### Output
- `checkpoints/backbone_synthetic.pth` - Backbone trained on synthetic data

---

## Stage 2: Real Fine-tuning

### Objective
Adapt the backbone to real camera domain (lighting, noise, colors).

### Configuration

```yaml
# config/stage2_real.yaml
model:
  backbone: efficientnet_b0
  pretrained: checkpoints/backbone_synthetic.pth  # From Stage 1
  num_classes: 105  # Set 45345-1 parts

data:
  sources:
    - real: 1700
  batch_size: 32
  image_size: 224
  sample_weight: 3.0  # Weight real samples higher

training:
  epochs: 30
  optimizer: AdamW
  learning_rate: 5e-5  # Lower LR for fine-tuning
  weight_decay: 0.01
  scheduler: CosineAnnealingWarmRestarts

augmentation:
  # Aggressive augmentation for small dataset
  - RandomRotate90
  - HorizontalFlip
  - ShiftScaleRotate(shift=0.1, scale=0.2, rotate=15)
  - RandomBrightnessContrast(brightness=0.3, contrast=0.3)
  - HueSaturationValue(hue=10, sat=30, val=30)
  - GaussNoise(var_limit=50)
  - CoarseDropout(max_holes=8)
```

### Multi-Task Architecture

```python
# scripts/training/stage2_real.py

class Stage2Model(nn.Module):
    def __init__(self, num_parts=105, num_colors=50):
        super().__init__()

        # Load Stage 1 backbone
        stage1 = Stage1Model(num_classes=200)
        stage1.load_state_dict(torch.load('checkpoints/backbone_synthetic.pth'))

        # Extract backbone (remove classifier)
        self.backbone = stage1.backbone.features
        self.avgpool = stage1.backbone.avgpool

        # New heads for fine-tuning
        self.shared = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.part_head = nn.Linear(512, num_parts)
        self.color_head = nn.Linear(512, num_colors)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.flatten(1)

        shared = self.shared(x)
        part_logits = self.part_head(shared)
        color_logits = self.color_head(shared)

        return part_logits, color_logits

    def get_embedding(self, x):
        """Extract 1280-dim embedding for Vector Space"""
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return nn.functional.normalize(x, dim=1)
```

### Output
- `checkpoints/backbone_final.pth` - Production-ready backbone
- `checkpoints/classifier_45345.pth` - Classifier head for set 45345-1

---

## Stage 3: Deployment Options

### Option A: Classifier (Demo - Set 45345-1)

```python
# For known set with fixed parts
class LegoClassifier:
    def __init__(self, model_path, part_mapping):
        self.model = Stage2Model(num_parts=105, num_colors=50)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.part_mapping = part_mapping

    def predict(self, image):
        with torch.no_grad():
            part_logits, color_logits = self.model(image)
            part_idx = part_logits.argmax(1)
            color_idx = color_logits.argmax(1)

        return {
            "part_id": self.part_mapping[part_idx],
            "color_id": self.color_mapping[color_idx],
            "confidence": softmax(part_logits).max()
        }
```

### Option B: Vector Space (Expansion)

```python
# For adding new sets without retraining
class LegoVectorSearch:
    def __init__(self, model_path, embeddings_path):
        self.model = Stage2Model(num_parts=105, num_colors=50)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Load pre-computed embeddings
        with open(embeddings_path, 'rb') as f:
            self.db = pickle.load(f)

        self.vectors = np.array([v['embedding'] for v in self.db.values()])
        self.keys = list(self.db.keys())

    def predict(self, image):
        with torch.no_grad():
            query_emb = self.model.get_embedding(image)

        # Cosine similarity search
        similarities = np.dot(self.vectors, query_emb.numpy())
        top_idx = np.argsort(similarities)[::-1][:5]

        return [
            {"part_id": self.db[self.keys[idx]]["part_id"],
             "confidence": similarities[idx]}
            for idx in top_idx
        ]

# Re-extract embeddings with trained backbone
def rebuild_embeddings(model, image_dir, output_path):
    """Use trained backbone to create better embeddings"""
    embeddings = {}
    for img_path in glob(f"{image_dir}/**/*.jpg"):
        img = load_and_preprocess(img_path)
        emb = model.get_embedding(img)
        embeddings[img_path] = {
            "embedding": emb.numpy(),
            "part_id": extract_part_id(img_path),
            "color_id": extract_color_id(img_path),
        }

    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
```

---

## Pi Deployment Analysis

### Memory Requirements

| Component | Size |
|-----------|------|
| EfficientNetB0 (ONNX, fp16) | ~10 MB |
| Rembg U2-Net | ~170 MB |
| Vector DB (94k, fp16) | ~230 MB |
| Python Runtime | ~200 MB |
| **Total** | **~610 MB** |

### Inference Time (Pi 5)

| Mode | Rembg | Model | Search | Total |
|------|-------|-------|--------|-------|
| **Pi-Only** | 2-4s | 0.3-0.5s | 0.05s | **2.5-5s** |
| **Pi+PC** | - | 0.1s (GPU) | 0.05s | **~0.5s** |

### Deployment Decision

| Scenario | Recommendation |
|----------|----------------|
| Demo / Offline | Pi-Only (slower but independent) |
| Production | Pi+PC (faster, scalable) |
| Low latency required | Pi+PC with GPU |

### Rembg Optimization (Future)

**Bottleneck**: Rembg (U2-Net) accounts for 80% of Pi inference time (2-4s out of 2.5-5s total).

**Observation**: Pi turntable captures have relatively clean, consistent backgrounds compared to arbitrary images.

**Potential Optimizations:**

| Approach | Inference Time | Trade-off |
|----------|----------------|-----------|
| U2-Net (current) | 2-4s | Best quality, slowest |
| HSV-based removal | ~0.1s | Fast, requires tuned thresholds |
| No removal | 0s | Fastest, requires training with turntable BG |
| MobileNet segmentation | ~0.3s | Balanced speed/quality |

**Recommendation for M4:**
- Stage 2 fine-tuning: Include both clean (rembg) and turntable backgrounds in training data
- This makes the model robust to background variations
- At deployment: Test accuracy without rembg; if acceptable (>90%), skip it for **~0.5s Pi-only inference**

---

## Timeline

| Week | Phase | Tasks |
|------|-------|-------|
| **1** | Stage 1 | Prepare B200C sampler, train synthetic backbone |
| **2** | Stage 2 | Fine-tune on real captures, evaluate |
| **3** | Stage 3A | Train classifier head for 45345-1, test accuracy |
| **3** | Stage 3B | Re-extract embeddings with trained backbone |
| **4** | Deploy | Export ONNX, test on Pi, production setup |

---

## Files to Create

```
scripts/
├── training/
│   ├── stage1_synthetic.py    # Synthetic pre-training
│   ├── stage2_real.py         # Real fine-tuning
│   ├── export_onnx.py         # Model export
│   └── rebuild_embeddings.py  # Re-extract with trained backbone
│
├── data/
│   ├── prepare_b200c.py       # Sample & upscale B200C
│   └── prepare_dataset.py     # Create train/val/test splits

config/
├── stage1_synthetic.yaml
├── stage2_real.yaml
└── part_mappings/
    └── 45345-1.json           # Part ID → index mapping

models/
├── backbone_synthetic.pth     # Stage 1 output
├── backbone_final.pth         # Stage 2 output
├── classifier_45345.onnx      # Exported classifier
└── trained_embeddings.pkl     # Re-extracted embeddings
```

---

## Success Criteria

| Metric | Target | Validation |
|--------|--------|------------|
| Part Top-1 (Classifier) | >95% | Real test set |
| Part Top-5 (Classifier) | >99% | Real test set |
| Part Top-1 (Vector) | >90% | Real test set |
| Color Accuracy | >85% | Real test set (where applicable) |
| Inference Time (Pi) | <5s | Pi 5 standalone |
| Inference Time (Pi+PC) | <1s | Pi capture + PC inference |

---

## Key Insights

1. **B200C 64×64 is usable** - Lanczos upscale + blur augmentation bridges the gap
2. **Three-stage training** - Synthetic pre-train → Real fine-tune → Deploy
3. **Dual deployment** - Classifier for accuracy, Vector for expansion
4. **Trained backbone benefits both** - Better features improve vector search too
5. **Pi-standalone is possible** - 2.5-5s latency, suitable for demo/offline use
6. **Rembg is the bottleneck** - Pi turntable has clean background; may skip rembg for ~0.5s inference
