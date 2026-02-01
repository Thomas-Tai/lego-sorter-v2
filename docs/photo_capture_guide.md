# Photo Capture Implementation Guide

This guide explains how to set up the Raspberry Pi environment and USB camera for the LEGO Sorter image acquisition system.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Raspberry Pi Setup](#raspberry-pi-setup)
- [Camera Setup](#camera-setup)
- [Deployment](#deployment)
- [Running the Image Acquirer](#running-the-image-acquirer)
- [Troubleshooting](#troubleshooting)

---

## Overview

The photo capture system is designed to automatically capture multi-angle images of LEGO parts for training the classification model. It uses:

- **USB Camera**: Standard USB webcam connected to the Raspberry Pi
- **Turntable**: Motorized platform to rotate parts for multi-angle capture
- **LED Lighting**: Controlled lighting for consistent image quality

The system captures **8 images per part** at 45-degree intervals, providing comprehensive coverage for model training.

---

## Architecture

The implementation follows a clean architecture with dependency injection:

```
┌─────────────────────────────────────────────────────────────┐
│                      ImageAcquirer                          │
│  (Orchestrates the capture workflow)                        │
├─────────────────────────────────────────────────────────────┤
│                           │                                 │
│         ┌─────────────────┼─────────────────┐               │
│         ▼                 ▼                 ▼               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
│  │  Hardware   │   │   Vision    │   │  Database   │        │
│  │  Service    │   │   Service   │   │  (SQLite)   │        │
│  └─────────────┘   └─────────────┘   └─────────────┘        │
│         │                 │                                 │
│         ▼                 ▼                                 │
│  ┌─────────────┐   ┌─────────────┐                          │
│  │  GPIO Pins  │   │ USB Camera  │                          │
│  │  (LEDs,     │   │ (OpenCV)    │                          │
│  │   Motor)    │   │             │                          │
│  └─────────────┘   └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `scripts/acquirer/run_acquisition.py` | Entry point for Acquirer Pi |
| `modules/hardware/motor.py` | MotorDriver class |
| `modules/hardware/led.py` | LedDriver class |
| `modules/hardware/camera.py` | CameraDriver class (OpenCV wrapper) |
| `modules/acquisition/acquirer.py` | ImageAcquirer orchestrator |
| `modules/database/manager.py` | Database operations |
| `config/hardware.yaml` | Hardware configuration settings |

---

## Raspberry Pi Setup

### Prerequisites

- Raspberry Pi 4 (recommended) or Pi 3B+
- Raspberry Pi OS (Bullseye or later)
- USB Camera (any V4L2-compatible webcam)
- Network connection (for initial setup)

### Automated Setup

1. **Deploy the code** from your Windows machine:
   ```powershell
   cd CodeBase\lego-sorter-v2
   .\scripts\acquirer\deploy.ps1
   ```

2. **SSH into the Pi**:
   ```bash
   ssh legoSorter
   ```

3. **Run the setup script** (first time only):
   ```bash
   bash ~/lego-sorter-v2/scripts/acquirer/setup_pi.sh
   ```

4. **Log out and log back in** (required for camera permissions):
   ```bash
   exit
   ssh legoSorter
   ```

### Manual Setup (Alternative)

If you prefer to set up manually:

```bash
# Update system
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libopencv-dev swig python3-dev liblgpio-dev

# Add user to video group (for camera access)
sudo usermod -aG video $USER

# Create virtual environment
python3 -m venv ~/lego-sorter-env
source ~/lego-sorter-env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r ~/lego-sorter-v2/requirements.txt
```

---

## Camera Setup

### Supported Cameras

Any USB camera compatible with Video4Linux2 (V4L2) should work:
- Logitech C920/C922
- Microsoft LifeCam
- Generic USB webcams

### Verifying Camera Connection

1. **Check if camera is detected**:
   ```bash
   ls /dev/video*
   ```
   You should see `/dev/video0` (or similar).

2. **Test camera with Python**:
   ```bash
   source ~/lego-sorter-env/bin/activate
   python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'FAIL'); cap.release()"
   ```

3. **Capture a test image**:
   ```bash
   python -c "
   import cv2
   cap = cv2.VideoCapture(0)
   ret, frame = cap.read()
   if ret:
       cv2.imwrite('/tmp/test.jpg', frame)
       print('Image saved to /tmp/test.jpg')
   cap.release()
   "
   ```

### Configuration

Camera settings are defined in `config/hardware.yaml`:

```yaml
camera:
  device_index: 0  # USB camera device index (/dev/video0)
```

If you have multiple cameras, change `device_index` to match your camera.

---

## Deployment

### Initial Deployment

From your Windows development machine:

```powershell
# Navigate to the project
cd CodeBase\lego-sorter-v2

# Normal deploy (incremental, hash-checks database)
.\scripts\acquirer\deploy.ps1

# Fresh deploy (removes old code folders on Pi first)
.\scripts\acquirer\deploy.ps1 -Clean
```

This script:
1. Creates the project directory on the Pi
2. Copies source code, config, and requirements
3. Uses hash check to skip unchanged database (~4 sec vs 4+ min transfer)
4. Displays next steps

### Updating Code

Run the same deploy script to push updates:

```powershell
.\scripts\acquirer\deploy.ps1
```

> **Note**: The `-Clean` flag removes `modules/`, `scripts/`, `config/` before deploying but preserves `data/` (images and database).

---

## Hardware Verification (Smoke Test)

Before running the main application, verify your soldering and wiring using the dedicated hardware check script.

### Running the Test
**WARNING**: This test activates the 12V motor and LED rail. Ensure your external power supply is connected.

1. SSH into the Pi and activate the environment:
   ```bash
   ssh legoSorter
   source ~/lego-sorter-env/bin/activate
   ```

2. Run the check script:
   ```bash
   python ~/lego-sorter-v2/scripts/hardware_check.py
   ```

### Verification Steps
Follow the on-screen prompts. The script is interactive:

1.  **LED Check (Pin 23)**:
    -   Script will blink the LED 3 times (full brightness).
    -   Then performing a **Gamma-Corrected Fade** (0% -> 100% -> 0%).
    -   *Pass Criteria*: Light should fade smoothly without flickering.

2.  **Motor Check (Pins 13, 15, 19, 21)**:
    -   Script will rotate the motor Clockwise for 5 seconds using an 8-step sequence.
    -   *Pass Criteria*: Shaft rotates steadily with torque. **Crucially**, after stopping, the shaft should spin freely (coils de-energized).

3.  **Button Check (Pin 11)**:
    -   Script will wait up to 10 seconds for a press.
    -   *Action*: Press the silver button when prompted.
    -   *Pass Criteria*: Script prints `>> SUCCESS: Button Press Detected!`. If you don't press it, it must timeout with `>> TIMEOUT`.

---

## Running the Image Acquirer

### Basic Usage

1. **Activate the virtual environment**:
   ```bash
   source ~/lego-sorter-env/bin/activate
   ```

2. **Navigate to the project**:
   ```bash
   cd ~/lego-sorter-v2
   ```

3. **Run the acquirer**:
   ```bash
   python scripts/acquirer/run_acquisition.py --set 45345-1
   ```

### Workflow

The image acquirer will:

1. Query the database for parts without images
2. For each part:
   - Display a prompt asking you to place the part
   - Wait for you to press ENTER
   - Turn on the LEDs
   - Capture 6 images (rotating 60° between each)
   - Turn off the LEDs
   - Update the database

### Output

Images are saved in a **hierarchical structure** for flexible training:

```
data/images/
├── manifest.csv              # Full metadata for all images
└── raw/
    └── {part_num}/           # e.g., 3001/
        └── {color_id}/       # e.g., 15/ (White)
            ├── 3001_15_0.jpg
            ├── 3001_15_1.jpg
            ├── ...
            └── 3001_15_7.jpg  # 8 angles
```

**manifest.csv columns:**
| Column | Example |
|--------|--------|
| image_path | `raw/3001/15/3001_15_0.jpg` |
| part_num | `3001` |
| color_id | `15` |
| color_name | `White` |
| part_name | `Brick 2x4` |
| angle | `0` |
| timestamp | `2026-01-30T12:00:00` |

**Training flexibility:**
- **Part identification**: Group by `part_num` folder
- **Color identification**: Group by `color_id` subfolder
- **Combined**: Use full path or manifest

---

## Troubleshooting

### Camera Not Found

**Symptom**: `CameraError: Failed to open camera at index 0`

**Solutions**:
1. Check if camera is connected:
   ```bash
   ls /dev/video*
   ```
2. Verify user has video group permissions:
   ```bash
   groups
   # Should include 'video'
   ```
3. If you just added the video group, log out and back in.

### Permission Denied

**Symptom**: Cannot access `/dev/video0`

**Solution**:
```bash
sudo usermod -aG video $USER
# Log out and back in
```

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'cv2'`

**Solution**:
```bash
source ~/lego-sorter-env/bin/activate
pip install opencv-python-headless
```

### Camera Returns Black Images

**Symptom**: Images are saved but appear completely black

**Solutions**:
1. Some cameras need "warm-up" time. Try adding a delay before capture.
2. Check if the lens cap is removed.
3. Ensure adequate lighting.

### SSH Connection Issues

If you can't connect via `ssh legoSorter`, see the [SSH Setup Guide](ssh_setup.md).

---

## Development Notes

### Testing Without Hardware

The codebase includes mock services for testing without physical hardware:

```python
from tests.mocks.mock_hardware_service import MockHardwareService
from tests.mocks.mock_vision_service import MockVisionService

# Use in tests
hardware = MockHardwareService()
vision = MockVisionService()
acquirer = ImageAcquirer(db_path, output_path, hardware, vision)
```

### Running Tests

```bash
# On Windows (development machine)
.\venv\Scripts\python.exe -m pytest tests\ -v

# On Raspberry Pi
source ~/lego-sorter-env/bin/activate
python -m pytest tests/ -v
```

---

## Related Documentation

- [SSH Setup Guide](ssh_setup.md) – Setting up SSH connection to the Pi
- [Hardware Configuration](../config/hardware.yaml) – Camera and GPIO settings
- [Main README](../README.md) – Project overview and getting started
