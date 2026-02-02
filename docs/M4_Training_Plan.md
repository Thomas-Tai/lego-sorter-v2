# M4: Model Training Plan

## Objective
Train a LEGO part classifier that accurately identifies **part ID** and **color ID** from real camera captures, using all available data sources (legacy, b200c, real).

---

## Phase 1: Data Preparation

### 1.1 Data Inventory

| Source | Images | Parts | Colors | Quality |
|--------|--------|-------|--------|---------|
| Legacy | 82,354 | ~6,000 | 251 | Rendered, white BG |
| B200C | 10,000 | ~200 | None (9999) | Synthetic 3D, multi-angle |
| Real | 1,687 | ~50 | 22 | Camera captures |

### 1.2 Data Filtering

Focus on the **200 target parts** for the sorter:

```python
# Filter database to target parts only
target_parts = load_target_parts("config/target_parts.txt")  # 200 parts

filtered_data = {
    "legacy": [],   # ~200 parts × ~40 colors = ~8,000 images
    "b200c": [],    # ~200 parts × 50 views = ~10,000 images
    "real": []      # Need to capture more
}
```

### 1.3 Real Capture Requirements

**Current gap**: Only ~50 parts have real captures. Need 150+ more parts.

| Priority | Parts | Images Needed | Effort |
|----------|-------|---------------|--------|
| High | 50 most common | 8 angles × 3 colors = 1,200 | 2-3 days |
| Medium | 100 common | 8 angles × 2 colors = 1,600 | 3-4 days |
| Low | 50 rare | 8 angles × 1 color = 400 | 1-2 days |

**Total**: ~3,200 new real captures needed.

### 1.4 Data Augmentation Pipeline

```python
train_augmentation = A.Compose([
    # Geometric
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),

    # Color/Lighting (bridge domain gap)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),

    # Background variation
    A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.3),

    # Normalize
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

### 1.5 Train/Val/Test Split

```
Real captures:    70% train / 15% val / 15% test
Legacy + B200C:   100% train (domain adaptation)

Strategy: Validate and test ONLY on real captures to measure real-world performance.
```

---

## Phase 2: Model Architecture

### 2.1 Option A: Single Multi-Task Model (Recommended)

```
Input (224×224×3)
       ↓
┌──────────────────────────┐
│  EfficientNetB0          │
│  (pretrained ImageNet)   │
│  Freeze first 50% layers │
└──────────────────────────┘
       ↓
    Features (1280-dim)
       ↓
┌──────────────────────────┐
│  Shared FC (512)         │
│  BatchNorm + ReLU        │
│  Dropout(0.3)            │
└──────────────────────────┘
       ↓
   ┌───┴───┐
   ↓       ↓
┌──────┐ ┌──────┐
│Part  │ │Color │
│Head  │ │Head  │
│(200) │ │(50)  │
└──────┘ └──────┘
   ↓       ↓
Part ID  Color ID
```

### 2.2 Option B: Two Separate Models

```
Model 1: Part Classifier (200 classes)
- Use ALL data (legacy + b200c + real)
- Color-agnostic (grayscale augmentation)

Model 2: Color Classifier (50 classes)
- Use only legacy + real (have color labels)
- Part-agnostic (train on crops)
```

### 2.3 Model Code

```python
import torch
import torch.nn as nn
from torchvision import models

class LegoClassifier(nn.Module):
    def __init__(self, num_parts=200, num_colors=50):
        super().__init__()

        # Backbone
        self.backbone = models.efficientnet_b0(pretrained=True)
        backbone_out = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(backbone_out, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Task heads
        self.part_head = nn.Linear(512, num_parts)
        self.color_head = nn.Linear(512, num_colors)

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared(features)

        part_logits = self.part_head(shared)
        color_logits = self.color_head(shared)

        return part_logits, color_logits
```

---

## Phase 3: Training Strategy

### 3.1 Loss Function

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, part_weight=1.0, color_weight=0.5):
        super().__init__()
        self.part_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.color_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.part_weight = part_weight
        self.color_weight = color_weight

    def forward(self, part_pred, color_pred, part_target, color_target):
        loss_part = self.part_loss(part_pred, part_target)

        # Only compute color loss for samples with known color (not 9999)
        valid_color_mask = color_target != COLOR_UNKNOWN_IDX
        if valid_color_mask.sum() > 0:
            loss_color = self.color_loss(
                color_pred[valid_color_mask],
                color_target[valid_color_mask]
            )
        else:
            loss_color = 0.0

        return self.part_weight * loss_part + self.color_weight * loss_color
```

### 3.2 Domain Adaptation Strategy

To bridge the gap between synthetic (legacy/b200c) and real captures:

```python
# Source weighting: Real captures get higher weight
sample_weights = {
    "real": 3.0,    # 3x weight for real captures
    "legacy": 1.0,
    "b200c": 1.0
}

# Curriculum learning: Start with synthetic, gradually add real
epoch_schedule = {
    0-5:   {"real": 0.2, "legacy": 0.4, "b200c": 0.4},  # Mostly synthetic
    5-15:  {"real": 0.5, "legacy": 0.25, "b200c": 0.25}, # Balanced
    15-30: {"real": 0.7, "legacy": 0.15, "b200c": 0.15}  # Mostly real
}
```

### 3.3 Training Hyperparameters

```yaml
# config/training.yaml
model:
  backbone: efficientnet_b0
  pretrained: true
  freeze_backbone_ratio: 0.5

training:
  epochs: 30
  batch_size: 64
  optimizer: AdamW
  learning_rate: 1e-4
  weight_decay: 0.01
  scheduler: CosineAnnealingLR
  warmup_epochs: 3

augmentation:
  train: strong
  val: none

early_stopping:
  patience: 5
  monitor: val_part_accuracy
```

### 3.4 Training Script Outline

```python
# scripts/training/train_classifier.py

def train():
    # 1. Load data
    train_dataset = LegoDataset(split="train", augment=True)
    val_dataset = LegoDataset(split="val", augment=False)

    # 2. Create model
    model = LegoClassifier(num_parts=200, num_colors=50)
    model = model.to(device)

    # 3. Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    criterion = MultiTaskLoss()

    # 4. Training loop
    for epoch in range(30):
        # Train
        model.train()
        for batch in train_loader:
            images, part_labels, color_labels = batch
            part_pred, color_pred = model(images)
            loss = criterion(part_pred, color_pred, part_labels, color_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_metrics = evaluate(model, val_loader)

        # Save best
        if val_metrics["part_acc"] > best_acc:
            torch.save(model.state_dict(), "checkpoints/best_model.pth")

        scheduler.step()
```

---

## Phase 4: Evaluation

### 4.1 Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Part Top-1 Accuracy | >90% | Correct part in top prediction |
| Part Top-5 Accuracy | >98% | Correct part in top 5 |
| Color Accuracy | >85% | Correct color (when known) |
| Inference Time | <100ms | Per image on Pi |

### 4.2 Evaluation Script

```python
# scripts/training/evaluate.py

def evaluate(model, test_loader):
    model.eval()

    part_correct = 0
    part_top5_correct = 0
    color_correct = 0
    color_total = 0
    total = 0

    with torch.no_grad():
        for images, part_labels, color_labels in test_loader:
            part_pred, color_pred = model(images)

            # Part accuracy
            _, part_predicted = part_pred.max(1)
            part_correct += (part_predicted == part_labels).sum().item()

            # Part top-5
            _, part_top5 = part_pred.topk(5, dim=1)
            part_top5_correct += (part_top5 == part_labels.unsqueeze(1)).any(1).sum().item()

            # Color accuracy (only valid colors)
            valid_mask = color_labels != COLOR_UNKNOWN_IDX
            if valid_mask.sum() > 0:
                _, color_predicted = color_pred[valid_mask].max(1)
                color_correct += (color_predicted == color_labels[valid_mask]).sum().item()
                color_total += valid_mask.sum().item()

            total += part_labels.size(0)

    return {
        "part_acc": part_correct / total,
        "part_top5_acc": part_top5_correct / total,
        "color_acc": color_correct / color_total if color_total > 0 else 0
    }
```

### 4.3 Confusion Analysis

```python
# Identify commonly confused parts
def confusion_analysis(model, test_loader):
    # Build confusion matrix
    confusion = np.zeros((num_parts, num_parts))

    for images, part_labels, _ in test_loader:
        part_pred, _ = model(images)
        _, predicted = part_pred.max(1)

        for true, pred in zip(part_labels, predicted):
            confusion[true][pred] += 1

    # Find top confused pairs
    confused_pairs = []
    for i in range(num_parts):
        for j in range(num_parts):
            if i != j and confusion[i][j] > 5:
                confused_pairs.append((
                    part_names[i],
                    part_names[j],
                    confusion[i][j]
                ))

    return sorted(confused_pairs, key=lambda x: -x[2])
```

---

## Phase 5: Deployment

### 5.1 Model Export

```python
# Export to ONNX for Pi deployment
def export_onnx(model, output_path):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["part_logits", "color_logits"],
        dynamic_axes={"image": {0: "batch"}},
        opset_version=11
    )
```

### 5.2 Inference API Update

```python
# sorter_app/services/inference_api.py (updated)

class ClassifierLoader:
    def __init__(self):
        self.session = ort.InferenceSession("models/lego_classifier.onnx")
        self.part_names = load_json("config/part_names.json")
        self.color_names = load_json("config/color_names.json")

    def predict(self, image_bytes):
        # Preprocess
        img = preprocess_image(image_bytes)

        # Inference
        part_logits, color_logits = self.session.run(
            None, {"image": img}
        )

        # Decode
        part_idx = np.argmax(part_logits)
        color_idx = np.argmax(color_logits)

        return {
            "part_id": self.part_names[part_idx],
            "part_confidence": softmax(part_logits)[part_idx],
            "color_id": self.color_names[color_idx],
            "color_confidence": softmax(color_logits)[color_idx]
        }
```

### 5.3 Pi Deployment

```bash
# On Raspberry Pi
pip install onnxruntime  # CPU version for Pi

# Copy model
scp models/lego_classifier.onnx pi@lego-sorter.local:~/lego-sorter-v2/models/
```

---

## Timeline

| Week | Phase | Tasks |
|------|-------|-------|
| 1 | Data Prep | Filter 200 parts, setup augmentation pipeline |
| 1-2 | Capture | Photograph 50 high-priority parts (1,200 images) |
| 2 | Training Setup | Model code, loss functions, data loaders |
| 2-3 | Training V1 | Initial training, hyperparameter tuning |
| 3 | Evaluation | Confusion analysis, identify weak points |
| 3-4 | Capture More | Add 100 medium-priority parts |
| 4 | Training V2 | Retrain with expanded dataset |
| 4 | Deployment | Export ONNX, update inference API, test on Pi |

**Total: ~4 weeks**

---

## Files to Create

```
scripts/
├── training/
│   ├── prepare_dataset.py     # Filter & organize training data
│   ├── train_classifier.py    # Main training script
│   ├── evaluate.py            # Evaluation metrics
│   └── export_onnx.py         # Model export

config/
├── training.yaml              # Hyperparameters
├── target_parts.json          # 200 target part IDs
├── part_names.json            # Part ID → index mapping
└── color_names.json           # Color ID → index mapping

models/
└── lego_classifier.onnx       # Exported model
```

---

## Success Criteria

- [ ] Part Top-1 Accuracy > 90% on real test set
- [ ] Part Top-5 Accuracy > 98% on real test set
- [ ] Color Accuracy > 85% on real test set
- [ ] Inference < 100ms on Raspberry Pi
- [ ] Works reliably for all 200 target parts
