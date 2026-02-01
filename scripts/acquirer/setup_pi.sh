#!/bin/bash
# Raspberry Pi Setup Script for Lego Sorter
# Run this script once after first deployment
#
# Usage: bash ~/lego-sorter-v2/scripts/setup_pi.sh

set -e  # Exit on any error

VENV_PATH="$HOME/lego-sorter-env"
PROJECT_PATH="$HOME/lego-sorter-v2"

echo "=== Lego Sorter Pi Setup ==="
echo ""

# System dependencies
echo "[1/5] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libopencv-dev swig python3-dev liblgpio-dev

# Camera permissions
echo ""
echo "[2/5] Configuring camera permissions..."
sudo usermod -aG video "$USER"

# Virtual environment
echo ""
echo "[3/5] Creating virtual environment..."
python3 -m venv "$VENV_PATH"

# Python dependencies
echo ""
echo "[4/5] Installing Python packages..."
source "$VENV_PATH/bin/activate"
pip install --upgrade pip
pip install -r "$PROJECT_PATH/requirements.txt"
pip install -r "$PROJECT_PATH/requirements-pi.txt"

# Verify installation
echo ""
echo "[5/5] Verifying installation..."
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "NOTE: Log out and back in for camera permissions to take effect."
echo ""
echo "To run the sorter:"
echo "  source $VENV_PATH/bin/activate"
echo "  cd $PROJECT_PATH"
echo "  # Run Acquirer (M2)"
echo "  python run_acquirer.py"
echo "  # Run Sorter (M3/M4)"
echo "  export LEGO_API_URL='http://<PC_IP>:8000'"
echo "  python sorter_app/main.py"

