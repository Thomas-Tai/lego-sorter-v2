# Lego Sorter V2

An AI-powered sorter for LEGO Spike Essential parts, built with a modular architecture for maintainability and dual-Pi deployment.

## 🎯 Project North Star

Deliver a commercial-grade, open-source LEGO sorting machine that serves as a comprehensive portfolio demonstrating full-stack engineering and disciplined project management.

## ✨ Features

- **Automated Sorting:** Sorts LEGO Spike Essential small parts with >85% accuracy target.
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
│   └── local/            # PC-only scripts
│       └── init_db.py    # Initialize database
│
├── data/                 # All data files
│   ├── raw/              # Rebrickable CSVs
│   ├── db/               # SQLite database
│   └── images/           # Captured training images
│       ├── raw/{part_num}/{color_id}/  # Hierarchical storage
│       └── manifest.csv  # Image metadata for training
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

### Training (PC)

```bash
python scripts/local/train_model.py
```
*(Coming in M4)*

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

### Guides
- [SSH Setup Guide](docs/ssh_setup.md)
- [Photo Capture Guide](docs/photo_capture_guide.md)