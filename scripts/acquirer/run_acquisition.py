#!/usr/bin/env python3
"""
Image Acquisition Script (Acquirer Pi Entry Point)
Run this on the Acquirer Pi to capture training images.
"""

import os
import sys
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.acquisition import ImageAcquirer

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Lego Sorter Image Acquisition Tool")
    parser.add_argument(
        "--set", type=str, default="45345-1", help="Set Number to process (default: Spike Essential 45345-1)"
    )
    args = parser.parse_args()

    print(f"Starting image acquisition for set: {args.set}")
    acquirer = ImageAcquirer(set_num=args.set)
    acquirer.run()


if __name__ == "__main__":
    main()
