import cv2
import numpy as np
import os

IMG_PATH = r"C:\Users\sky\.gemini\antigravity\brain\95db7288-0bbb-4b60-9e3f-17cbeb125dc2\camera_test_v5_final.jpg"


def main():
    if not os.path.exists(IMG_PATH):
        print(f"File not found: {IMG_PATH}")
        return

    img = cv2.imread(IMG_PATH)
    if img is None:
        print("Failed to load image.")
        return

    mean_val = np.mean(img)
    min_val = np.min(img)
    max_val = np.max(img)

    print(f"Image Stats:")
    print(f"  Dimensions: {img.shape}")
    print(f"  Mean Brightness: {mean_val:.2f} (0=Black, 255=White)")
    print(f"  Min Pixel: {min_val}")
    print(f"  Max Pixel: {max_val}")

    if mean_val < 5:
        print("CONCLUSION: Image is BLACK.")
    elif mean_val < 30:
        print("CONCLUSION: Image is VERY DARK (Underexposed).")
    else:
        print("CONCLUSION: Image has VISIBLE CONTENT.")


if __name__ == "__main__":
    main()
