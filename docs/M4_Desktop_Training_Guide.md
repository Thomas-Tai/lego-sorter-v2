# M4 Desktop Training Guide

## Target Machine
- **CPU**: Intel i7-13700F
- **GPU**: NVIDIA RTX 3060 (Ampere, sm_86) ✅ Supported by PyTorch
- **VRAM**: 12GB (sufficient for batch_size=64)

---

## Step 1: Files to Transfer

### Pre-Staged Directory (Ready to Copy)

All files have been prepared at `D:\LegoSorter` on the laptop. Copy this entire folder to the desktop.

```
D:\LegoSorter\                           # Total: ~133 GB
├── code\lego-sorter-v2\                 # 3.3 GB (includes embeddings/db)
│   ├── config\stage1_desktop.yaml       # Desktop-optimized config
│   ├── scripts\training\                # Training scripts
│   └── data\embeddings\                 # Hybrid DB vectors
├── data\
│   ├── b200c_processed\                 # 49 GB - 50,000 synthetic images (224×224)
│   └── legacy_images\                   # 81 GB - 82,354 legacy images
├── checkpoints\stage1\                  # Empty (for training outputs)
└── logs\stage1\                         # Empty (for TensorBoard)
```

### Transfer Options

**Option A: Direct Copy (Recommended)**
- Use external SSD/HDD to copy `D:\LegoSorter` folder (~133 GB)
- Or use network share if both machines are on same network

**Option B: Minimal (B200C only)**
- Copy only `code\` and `data\b200c_processed\` (~52 GB)
- Legacy images are optional for Stage 1 training

### Desktop Directory Structure

```
D:\LegoSorter\
├── code\lego-sorter-v2\     # Project code + embeddings
├── data\
│   ├── b200c_processed\     # 50,000 images (224×224)
│   └── legacy_images\       # 82,354 images
├── checkpoints\stage1\      # Training outputs
├── logs\stage1\             # TensorBoard logs
└── venv\                    # Python virtual environment (create on desktop)
```

---

## Step 2: Environment Setup (Desktop)

### 2.1 Install Python 3.10+

```powershell
# Download from python.org or use winget
winget install Python.Python.3.10
```

### 2.2 Create Virtual Environment

```powershell
cd D:/LegoSorter
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2.3 Install PyTorch with CUDA 12.1 (for RTX 3060)

```powershell
# RTX 3060 (Ampere sm_86) - use CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2.4 Verify GPU

```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Expected: CUDA: True, GPU: NVIDIA GeForce RTX 3060
```

### 2.5 Install Training Dependencies

```powershell
pip install albumentations scikit-learn tensorboard tqdm pyyaml opencv-python
```

---

## Step 3: Config (Already Set Up)

Desktop config `config/stage1_desktop.yaml` is already included with correct paths:

```yaml
data:
  legacy_dir: "D:/LegoSorter/data/legacy_images"
  b200c_processed_dir: "D:/LegoSorter/data/b200c_processed"

training:
  batch_size: 64          # Optimized for RTX 3060 12GB
  num_workers: 8          # Optimized for i7-13700F

output:
  checkpoint_dir: "D:/LegoSorter/checkpoints/stage1"
  log_dir: "D:/LegoSorter/logs/stage1"
```

No manual editing required if you copy `D:\LegoSorter` to the same path on desktop.

---

## Step 4: Run Training

### 4.1 Quick Test (1 epoch)

```powershell
cd D:/LegoSorter/code/lego-sorter-v2
python scripts/training/stage1_synthetic.py --config config/stage1_desktop.yaml --epochs 1
```

### 4.2 Full Training (15 epochs)

```powershell
python scripts/training/stage1_synthetic.py --config config/stage1_desktop.yaml
```

### Expected Performance (RTX 3060)

| Metric | Estimate |
|--------|----------|
| Batch size | 64 |
| Images/epoch | ~45,000 (90% train split) |
| Time/epoch | ~4-6 minutes |
| Total (15 epochs) | ~60-90 minutes |

---

## Step 5: Monitor Training

### TensorBoard

```powershell
# In another terminal
tensorboard --logdir D:/LegoSorter/logs/stage1
# Open http://localhost:6006
```

### Check GPU Usage

```powershell
nvidia-smi -l 1
# Should show ~8-10GB VRAM usage during training
```

---

## Step 6: Transfer Results Back

After training completes:

```
# Copy back to laptop/cloud
D:/LegoSorter/checkpoints/stage1/
├── backbone_synthetic.pth      # Best model (use for Stage 2)
├── backbone_synthetic_final.pth
└── checkpoint_epoch_*.pth

D:/LegoSorter/logs/stage1/
└── [TensorBoard logs]

D:/LegoSorter/code/lego-sorter-v2/metrics/
└── stage1_metrics.json
```

---

## Quick Reference Commands

```powershell
# === Desktop Setup (One-time) ===
cd D:/LegoSorter
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install albumentations scikit-learn tensorboard tqdm pyyaml opencv-python

# === Verify GPU ===
python -c "import torch; print(torch.cuda.get_device_name(0))"

# === Run Training ===
cd D:/LegoSorter/code/lego-sorter-v2
python scripts/training/stage1_synthetic.py --config config/stage1_desktop.yaml

# === Monitor (separate terminal) ===
tensorboard --logdir D:/LegoSorter/logs/stage1
```

---

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```powershell
python scripts/training/stage1_synthetic.py --batch-size 32
```

### Slow Training
- Check GPU utilization: `nvidia-smi`
- Increase num_workers in config (try 8 instead of 4)

### Import Errors
```powershell
pip install --upgrade albumentations torch torchvision
```

---

## Files Checklist

| Item | Size | Images | Status |
|------|------|--------|--------|
| Project code + embeddings | 3.3 GB | - | ✅ Ready at D:\LegoSorter |
| B200C processed | 49 GB | 50,000 | ✅ Ready at D:\LegoSorter |
| Legacy images | 81 GB | 82,354 | ✅ Ready at D:\LegoSorter |
| **Total** | **133 GB** | **132,354** | **Ready to transfer** |

---

## Next Steps After Stage 1

1. Transfer `backbone_synthetic.pth` back to laptop
2. Proceed with Stage 2 (Real Fine-tuning) - can run on desktop or laptop
3. Stage 2 uses only 1.7k images, much faster to train
