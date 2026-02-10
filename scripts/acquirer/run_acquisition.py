#!/usr/bin/env python3
"""
Image Acquisition Script (Acquirer Pi Entry Point)
Run this on the Acquirer Pi to capture training images.

Usage:
    python -m acquisition   (preferred)
    python scripts/acquirer/run_acquisition.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from acquisition.acquirer import main

if __name__ == "__main__":
    main()
