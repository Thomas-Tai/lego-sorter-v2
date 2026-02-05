"""
Script to sample colors (HSV) from an image.
"""

import sys
import cv2
import numpy as np


def analyze(path):
    print(f"Analyzing: {path}")
    img = cv2.imread(path)  # Loads as BGR
    if img is None:
        print("Failed to load.")
        return

    h, w = img.shape[:2]
    print(f"Size: {w}x{h}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Sample a grid
    grid_count = 5
    xs = np.linspace(0, w - 1, grid_count, dtype=int)
    ys = np.linspace(0, h - 1, grid_count, dtype=int)

    print("\nSampled Grid (HSV):")
    print("X\tY\tH\tS\tV")

    for y in ys:
        for x in xs:
            pixel = hsv[y, x]
            print(f"{x}\t{y}\t{pixel[0]}\t{pixel[1]}\t{pixel[2]}")


if __name__ == "__main__":
    analyze(sys.argv[1])
