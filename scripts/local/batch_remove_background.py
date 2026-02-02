"""
Batch Background Removal for Real Captures.

This script:
1. Scans data/images/raw/ for all Real Capture images.
2. Removes background using rembg (U-2-Net) with GPU acceleration.
3. Saves results to data/images/raw_clean/.
4. Maintains the same folder structure.
"""

import os
import sys
import time
from pathlib import Path
from PIL import Image
from rembg import remove as rembg_remove, new_session
import io

# Configuration
BACKGROUND_COLOR = (255, 255, 255)  # White background

# Create rembg session with CUDA provider for GPU acceleration
print("Initializing rembg with CUDA provider...")
SESSION = new_session(
    "u2net", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
print("rembg session ready.")


def process_image(input_path, output_path):
    """Remove background from a single image and save with white background."""
    try:
        # Read image bytes
        with open(input_path, "rb") as f:
            input_bytes = f.read()

        # Remove background (returns RGBA with transparent background)
        output_bytes = rembg_remove(input_bytes, session=SESSION)

        # Convert to PIL Image
        img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

        # Create white background
        background = Image.new("RGBA", img.size, BACKGROUND_COLOR + (255,))

        # Composite: paste image on white background
        composite = Image.alpha_composite(background, img)

        # Convert to RGB and save
        rgb_img = composite.convert("RGB")

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        rgb_img.save(output_path, format="JPEG", quality=95)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main():
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    raw_dir = os.path.join(project_root, "data", "images", "raw")
    clean_dir = os.path.join(project_root, "data", "images", "raw_clean")

    if not os.path.exists(raw_dir):
        print(f"Raw directory not found: {raw_dir}")
        return

    # Find all images
    print(f"Scanning {raw_dir}...")
    images = []
    for root, dirs, files in os.walk(raw_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                input_path = os.path.join(root, filename)
                # Calculate relative path
                rel_path = os.path.relpath(input_path, raw_dir)
                output_path = os.path.join(clean_dir, rel_path)
                images.append((input_path, output_path))

    print(f"Found {len(images)} images to process.")

    if not images:
        return

    # Process images
    start_time = time.time()
    success = 0
    failed = 0
    skipped = 0

    for i, (input_path, output_path) in enumerate(images):
        # Skip if already processed
        if os.path.exists(output_path):
            skipped += 1
            continue

        if process_image(input_path, output_path):
            success += 1
        else:
            failed += 1

        # Progress update
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(images) - i - 1) / rate if rate > 0 else 0

        sys.stdout.write(
            f"\rProgress: {i+1}/{len(images)} | "
            f"New: {success} | Failed: {failed} | Skipped: {skipped} | "
            f"Rate: {rate:.1f} img/s | ETA: {remaining:.0f}s"
        )
        sys.stdout.flush()

    print(f"\n\nDone! New: {success}, Failed: {failed}, Skipped: {skipped}")
    print(f"Output saved to: {clean_dir}")


if __name__ == "__main__":
    main()
