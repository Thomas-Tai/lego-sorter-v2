import os
import sys

# Add NVIDIA DLLs to PATH (Windows fix for onnxruntime-gpu)
# Must be done BEFORE importing onnxruntime or rembg
try:
    if os.name == "nt":
        # Dynamic path relative to current environment
        # sys.prefix points to the venv root (e.g. .../venv-gpu)
        venv_root = sys.prefix
        site_packages = os.path.join(venv_root, "Lib", "site-packages")
        nvidia_dir = os.path.join(site_packages, "nvidia")

        print(f"Searching for NVIDIA DLLs in: {nvidia_dir}")

        for lib in ["cudnn", "cublas", "cudart"]:
            bin_path = os.path.join(nvidia_dir, lib, "bin")
            if os.path.exists(bin_path):
                # Method 1: Python 3.8+
                os.add_dll_directory(bin_path)
                # Method 2: classic PATH (redundant but safe)
                os.environ["PATH"] += os.pathsep + bin_path
                print(f"Added DLL directory: {bin_path}")
except Exception as e:
    print(f"Warning: Failed to add NVIDIA DLLs: {e}")

from rembg import remove, new_session
from PIL import Image
import cv2
import numpy as np
import io
import time
from tqdm import tqdm

# Configuration
HEAVY_ASSETS_ROOT = r"C:\D\WorkSpace\[Local]_Station\01_Heavy_Assets\LegoSorterProject"
INPUT_ROOT = os.path.join(HEAVY_ASSETS_ROOT, "Data", "images", "raw")
OUTPUT_ROOT = os.path.join(HEAVY_ASSETS_ROOT, "Data", "images", "raw_clean")

# Filters (Tuned)
# Turntable Grey: S=[0-25], V=[100-160]
MASK_LOW = np.array([0, 0, 100])
MASK_HIGH = np.array([180, 25, 160])


def process_image(input_path, output_path, session):
    # 1. Load Image
    with open(input_path, "rb") as f:
        img_bytes = f.read()

    # 2. Rembg
    try:
        output_bytes = remove(img_bytes, session=session, alpha_matting=True)
        img_pil = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        img_np = np.array(img_pil)
    except Exception as e:
        print(f"Rembg failed for {input_path}: {e}")
        return False

    # 3. Color Filter
    rgb_img = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv_img, MASK_LOW, MASK_HIGH)

    # Apply color mask (make grey pixels transparent)
    img_np[mask == 255, 3] = 0

    # 4. Connected Component Analysis (Island Removal)
    alpha = img_np[:, :, 3]
    _, binary = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Keep largest
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(binary)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        img_np[clean_mask == 0, 3] = 0
    else:
        # If no contours found (empty image?), skip saving or save empty
        pass  # Image might be empty

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    res_pil = Image.fromarray(img_np)
    res_pil.save(output_path)
    return True


def main():
    print(f"Cleaning Raw Images...")
    print(f"Input: {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")

    session = new_session("u2net")

    # Collect files
    files_to_process = []
    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                files_to_process.append(os.path.join(root, file))

    print(f"Found {len(files_to_process)} images.")

    skipped = 0
    processed = 0
    failed = 0

    pbar = tqdm(files_to_process)
    for input_path in pbar:
        try:
            rel_path = os.path.relpath(input_path, INPUT_ROOT)

            # Change extension to .png for transparency
            base_rel = os.path.splitext(rel_path)[0] + ".png"
            output_path = os.path.join(OUTPUT_ROOT, base_rel)

            # Check if exists (Optional: Skip if exists to resume?)
            # For now, overwrite as per user request to "clean"

            success = process_image(input_path, output_path, session)
            if success:
                processed += 1
            else:
                failed += 1

            pbar.set_description(f"Processed: {processed} | Failed: {failed}")

        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            failed += 1

    print(f"\nDone. Processed: {processed}, Failed: {failed}")


if __name__ == "__main__":
    main()
