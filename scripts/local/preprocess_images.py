#!/usr/bin/env python
"""
Preprocess LEGO part images - crop to turntable area.

Usage:
    python scripts/local/preprocess_images.py
    python scripts/local/preprocess_images.py --size 224

Requirements:
    pip install opencv-python
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.training.preprocessing import preprocess_dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess LEGO part images - crop to turntable area"
    )
    parser.add_argument(
        "--raw",
        default="data/images/raw",
        help="Path to raw images (default: data/images/raw)",
    )
    parser.add_argument(
        "--output",
        default="data/images/processed",
        help="Path for processed images (default: data/images/processed)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=400,
        help="Output image size in pixels (square, default: 400)",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("LEGO Image Preprocessor")
    print("=" * 50)
    print(f"Raw images:    {args.raw}")
    print(f"Output:        {args.output}")
    print(f"Target size:   {args.size}x{args.size}")
    print("=" * 50)
    print()

    stats = preprocess_dataset(args.raw, args.output, args.size)

    print()
    print("=" * 50)
    print(f"Done! Processed images saved to: {args.output}")


if __name__ == "__main__":
    main()
