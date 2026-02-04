# Lego Sorter V2

An AI-powered sorter for LEGO Spike Essential parts, built with a modular architecture for maintainability and dual-Pi deployment.

## 🎯 Project North Star

Deliver a commercial-grade, open-source LEGO sorting machine that serves as a comprehensive portfolio demonstrating full-stack engineering and disciplined project management.

## ✨ Features

- **Automated Sorting:** Sorts LEGO Spike Essential small parts with **96% accuracy** (M4 Classifier).
- **Modular Architecture:** Independent modules for database, hardware, acquisition, and training.
- **Dual-Pi Setup:** Separate Pis for data collection (Acquirer) and sorting (Sorter).
- **Rebrickable Integration:** Imports official LEGO parts data for accurate identification.
- **Smart Deployment:** Hash-based sync skips unchanged files; optional clean deploy.

## 📁 Project Structure

```
lego-sorter-v2/
├── modules/              # Independent, reusable modules
│   ├── database/         # DatabaseManager, DataImporter
│   ├── hardware/         # MotorDriver, LedDriver, CameraDriver
│   ├── acquisition/      # ImageAcquirer
│   └── training/         # AI model training (M4)
│
├── scripts/              # Entry points & deployment
│   ├── acquirer/         # Acquirer Pi scripts
│   │   ├── deploy.ps1    # Deploy to Pi (supports -Clean flag)
│   │   ├── sync_data.ps1 # Sync images back to PC
│   │   ├── run_acquisition.py
│   │   └── setup_pi.sh   # Initial Pi setup
│   ├── sorter/           # Sorter Pi scripts
│   │   ├── deploy.ps1    # Deploy sorter to Pi
│   │   ├── run_sorter.sh # Run sorter with env config
│   │   └── sorter.env.template  # Environment config template
│   └── local/            # PC-only scripts (training, inference, DB init)
│
├── data/                 # All data files
│   ├── raw/              # Rebrickable CSVs
│   ├── db/               # SQLite database
│   ├── embeddings/       # Vector Database
│   │   ├── legacy_embeddings.pkl  # Processed legacy data
│   │   └── hybrid_embeddings.pkl  # Merged DB (Legacy + Real + B200C)
│   └── images/           # Captured training images
│       ├── raw/{part_num}/{color_id}/  # Hierarchical storage
│       ├── raw_clean/    # Processed (BG Removed) Real images
│       └── b200c_processed/ # Processed B200C synthetic data
│       └── manifest.csv  # Image metadata for training
│
├── models/               # Trained models
│   ├── lego_classifier.onnx   # M4 ONNX classifier (94 parts, 22 colors)
│   ├── part_mapping.json      # Part ID → class index mapping
│   └── color_mapping.json     # Color ID → class index mapping
│
├── config/               # Configuration files
├── tests/                # Unit tests
└── Project_Manage/       # Project documentation
```

> [!NOTE]
> **Large Asset Storage**: Virtual environments, datasets, and models are stored locally (non-synced) at `[Local]_Station/01_Heavy_Assets/LegoSorterProject`. See [Environment Setup](Project_Manage/environment_setup.md).

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Git
- (For Pi) Raspberry Pi OS with GPIO access

### Hardware Setup

- [SSH Setup Guide](docs/ssh_setup.md) - Raspberry Pi connection
- [Photo Capture Guide](docs/photo_capture_guide.md) - Camera and acquisition

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Thomas-Tai/lego-sorter-v2.git
    cd lego-sorter-v2
    ```

2.  **Create virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Mac/Linux
    python3.11 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Data Setup

1.  **Download Rebrickable Data:**
    - Go to [https://rebrickable.com/downloads/](https://rebrickable.com/downloads/)
    - Download: `sets.csv.gz`, `inventories.csv.gz`, `inventory_parts.csv.gz`, `parts.csv.gz`, `colors.csv.gz`

2.  **Prepare Data:**
    - Unzip all `.gz` files
    - Place CSVs in `data/raw/rebrickable_YYYYMMDD/`

3.  **Initialize Database:**
    ```bash
    python scripts/local/init_db.py
    ```
    This creates `data/db/lego_parts.sqlite`.

4.  **Data Pipeline Status (Current):**
    -   **Legacy Data**: ~82k entries (Processed)
    -   **Real Captures**: ~1.7k entries (Processed & Merged)
    -   **B200C Dataset**: ~10k entries (Synthetic, Merged)
    -   **Total Hybrid DB**: ~94k vectors (`data/embeddings/hybrid_embeddings.pkl`)
    -   **Source Weighting**: Enabled (Legacy 0.85×, B200C 0.95× penalty to prefer real captures)

    **Set 45345-1 Coverage:**
    -   105/116 parts have real captures (90% coverage)
    -   All sortable parts covered; only large components (hub, motors) missing

    **Verification Tools:**
    -   **Camera Inference Test**:
        ```bash
        python scripts/local/run_camera_inference.py
        ```
        *Note: On Windows, this script uses DirectShow (CAP_DSHOW) and MJPG to prevent black screen issues.*

    -   **Classification Debug**:
        ```bash
        python scripts/local/debug_classification.py
        ```

## 🔧 Usage

### Data Collection (Acquirer Pi)

1.  **Deploy to Pi:**
    ```powershell
    .\scripts\acquirer\deploy.ps1           # Normal deploy (incremental)
    .\scripts\acquirer\deploy.ps1 -Clean    # Fresh deploy (clears old code)
    ```
    > Hash check skips unchanged database transfers (~4 sec check vs 4+ min transfer).

2.  **First-time Pi setup:**
    ```bash
    ssh legoSorter
    bash ~/lego-sorter-v2/scripts/acquirer/setup_pi.sh
    # Log out and back in for camera permissions
    ```

3.  **Run acquisition:**
    ```bash
    source ~/lego-sorter-env/bin/activate
    cd ~/lego-sorter-v2
    python scripts/acquirer/run_acquisition.py --set 45345-1
    ```

4.  **Sync data back to PC:**
    ```powershell
    .\scripts\acquirer\sync_data.ps1
    ```

### Image Storage Format

Images are saved hierarchically for flexible training:
```
data/images/
├── manifest.csv                    # part_num, color_id, color_name, angle, timestamp
└── raw/
    └── {part_num}/
        └── {color_id}/
            ├── {part_num}_{color_id}_0.jpg
            ├── {part_num}_{color_id}_1.jpg
            └── ... (8 angles)
```

### Sorting (Sorter Pi)

1.  **Start Inference API on PC:**
    ```bash
    python -m uvicorn sorter_app.services.inference_api:app --host 0.0.0.0 --port 8000
    ```

2.  **Run sorter on Pi:**
    ```bash
    ssh legoSorter
    cd ~/lego-sorter-v2
    source venv/bin/activate
    export LEGO_API_URL=http://<PC_IP>:8000
    python -m sorter_app.main
    ```

### M4 Model Training (PC)

The M4 classifier achieves **96% Top-1 accuracy** on real LEGO parts.

-   **Architecture:** EfficientNet-B0 backbone + Part Head (94 classes) + Color Head (22 classes)
-   **Model:** `models/lego_classifier.onnx` (19 MB)
-   **Training Guide:** [M4 Desktop Training Guide](docs/M4_Desktop_Training_Guide.md)

```bash
# Train the model (requires GPU)
python scripts/training/stage2_real.py --config config/stage2_real.yaml

# Export to ONNX
python scripts/training/export_onnx.py
```

## 🧪 Development

Install dev tools:
```bash
pip install -r requirements-dev.txt
```

Run tests:
```bash
python -m pytest
```

## 📚 Documentation

### Project Management
- [Project Index](Project_Manage/README.md) - Documentation index
- [Technical Specification](Project_Manage/spec.md)
- [Requirements](Project_Manage/requirements.md)
- [Current Task Status](Project_Manage/task.md)
- [Decision Log](Project_Manage/decisions/DECISION_LOG.md)

### Technical Reports
- [AI Model Documentation](Project_Manage/reports/AI_Model_Technical_Documentation.md)
- [AI Plan Review](Project_Manage/reports/AI_Plan_Review.md)
- [Legacy Data Strategy Report](Project_Manage/reports/Legacy_Data_Strategy_Report_20260131.md)
- [Inference Verification](Project_Manage/reports/walkthrough_inference_verification.md)
- [API Security Audit](Project_Manage/security_audit_2026-02-01.md)
- [Environment & Asset Setup](Project_Manage/environment_setup_2026-02-01.md)

### Development
- [CI/CD Checklist](Project_Manage/CI_Checklist.md) - Pre-push verification guide
- [Pre-Push Script](scripts/pre_push_check.ps1) - Automated CI checks

### Guides
- [SSH Setup Guide](docs/ssh_setup.md)
- [Photo Capture Guide](docs/photo_capture_guide.md)
- [M4 Training Plan](docs/M4_Training_Plan.md) - Classifier training for set 45345-1
- [M4 Desktop Training Guide](docs/M4_Desktop_Training_Guide.md) - Step-by-step training on Windows

### Debug Tools
- `scripts/local/debug_classification.py` - Diagnose classification issues
- `scripts/local/run_camera_inference.py` - Real-time camera inference test