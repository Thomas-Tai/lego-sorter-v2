import cv2
import numpy as np
import os

# The image uploaded by the user
IMG_PATH = r"C:\Users\sky\.gemini\antigravity\brain\95db7288-0bbb-4b60-9e3f-17cbeb125dc2\uploaded_media_1770017698563.png"


def main():
    if not os.path.exists(IMG_PATH):
        print(f"File not found: {IMG_PATH}")
        return

    img = cv2.imread(IMG_PATH)
    if img is None:
        print("Failed to load image.")
        return

    # 1. Texture Analysis (Laplacian Variance) - Blur detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Histogram Analysis (Check for noise/static)
    # Static often has a very flat histogram or specific evenly distributed peaks
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_std = np.std(hist)

    print(f"Image Analysis:")
    print(f"  Laplacian Variance (Sharpness): {laplacian_var:.2f}")
    print(f"  Histogram Std Dev: {hist_std:.2f}")

    # Heuristics
    if laplacian_var > 3000:  # Extremely high edge energy often means static/noise
        print("CONCLUSION: Likely STATIC/NOISE (Variance too high for natural image).")
    elif laplacian_var < 10:
        print("CONCLUSION: Likely BLURRED or FLAT field.")
    else:
        print("CONCLUSION: Likely VALID image content.")


if __name__ == "__main__":
    main()
