"""
Process B200C LEGO Classification Dataset.

This script:
1. Samples N evenly-spaced views per part from the B200C dataset.
2. Applies rembg background removal with GPU acceleration.
3. Upscales from 64x64 to 224x224 for compatibility with EfficientNet.
4. Saves processed images to data/images/b200c_processed/.

Usage:
    python process_b200c_dataset.py [--views 50] [--skip-rembg]
"""

import os
import sys
import time
import argparse
from pathlib import Path
from PIL import Image
from rembg import remove as rembg_remove, new_session
import io

# Configuration
B200C_ROOT = r"C:\D\WorkSpace\[Local]_Station\01_Heavy_Assets\LegoSorterProject\Data\datasets\B200C LEGO Classification Dataset\64"
OUTPUT_DIR = r"C:\D\WorkSpace\[Local]_Station\01_Heavy_Assets\LegoSorterProject\Data\images\b200c_processed"
OUTPUT_SIZE = (224, 224)
BACKGROUND_COLOR = (255, 255, 255)  # White background


def get_output_dir():
    """Get the output directory path."""
    return OUTPUT_DIR


def sample_views(total_images: int, num_views: int) -> list:
    """Generate evenly-spaced indices to sample from total_images."""
    if num_views >= total_images:
        return list(range(total_images))
    step = total_images / num_views
    return [int(i * step) for i in range(num_views)]


def process_image_rembg(input_path: str, session) -> Image.Image:
    """Remove background and upscale image."""
    with open(input_path, "rb") as f:
        input_bytes = f.read()

    # Remove background
    output_bytes = rembg_remove(input_bytes, session=session)

    # Convert to PIL Image
    img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    # Create white background
    background = Image.new("RGBA", img.size, BACKGROUND_COLOR + (255,))
    composite = Image.alpha_composite(background, img)
    rgb_img = composite.convert("RGB")

    # Upscale to 224x224
    rgb_img = rgb_img.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)

    return rgb_img


def process_image_simple(input_path: str) -> Image.Image:
    """Simply upscale image without background removal."""
    img = Image.open(input_path).convert("RGB")
    img = img.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
    return img


def main():
    parser = argparse.ArgumentParser(description="Process B200C dataset")
    parser.add_argument("--views", type=int, default=50, help="Number of views to sample per part")
    parser.add_argument("--skip-rembg", action="store_true", help="Skip background removal")
    args = parser.parse_args()

    num_views = args.views
    use_rembg = not args.skip_rembg

    print(f"B200C Dataset Processor")
    print(f"=" * 50)
    print(f"Views per part: {num_views}")
    print(f"Background removal: {'Enabled' if use_rembg else 'Disabled'}")
    print(f"Output size: {OUTPUT_SIZE}")

    # Initialize rembg session
    session = None
    if use_rembg:
        print("\nInitializing rembg with CUDA provider...")
        try:
            session = new_session("u2net", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            print("rembg session ready (GPU accelerated).")
        except Exception as e:
            print(f"Warning: CUDA not available, falling back to CPU: {e}")
            session = new_session("u2net", providers=["CPUExecutionProvider"])

    # Check B200C path
    if not os.path.exists(B200C_ROOT):
        print(f"Error: B200C dataset not found at {B200C_ROOT}")
        return

    # Get output directory
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Scan part folders
    print(f"\nScanning {B200C_ROOT}...")
    part_folders = sorted([f for f in os.listdir(B200C_ROOT) if os.path.isdir(os.path.join(B200C_ROOT, f))])
    print(f"Found {len(part_folders)} parts.")

    # Calculate total images
    total_to_process = len(part_folders) * num_views
    print(f"Total images to process: {total_to_process}")

    # Process each part
    start_time = time.time()
    processed = 0
    skipped = 0
    failed = 0

    for part_idx, part_id in enumerate(part_folders):
        part_path = os.path.join(B200C_ROOT, part_id)

        # List all images in part folder
        images = sorted([f for f in os.listdir(part_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        if not images:
            continue

        # Sample view indices
        view_indices = sample_views(len(images), num_views)

        for view_idx in view_indices:
            if view_idx >= len(images):
                continue

            input_filename = images[view_idx]
            input_path = os.path.join(part_path, input_filename)

            # Output filename: partid_viewindex.jpg
            output_filename = f"{part_id}_v{view_idx:04d}.jpg"
            output_path = os.path.join(output_dir, output_filename)

            # Skip if already processed
            if os.path.exists(output_path):
                skipped += 1
                processed += 1
                continue

            try:
                if use_rembg:
                    img = process_image_rembg(input_path, session)
                else:
                    img = process_image_simple(input_path)

                img.save(output_path, format="JPEG", quality=95)
                processed += 1

            except Exception as e:
                failed += 1
                if failed <= 5:  # Only show first 5 errors
                    print(f"\nError processing {input_path}: {e}")

        # Progress update
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (total_to_process - processed) / rate if rate > 0 else 0

        sys.stdout.write(
            f"\rParts: {part_idx + 1}/{len(part_folders)} | "
            f"Images: {processed}/{total_to_process} | "
            f"Skipped: {skipped} | Failed: {failed} | "
            f"Rate: {rate:.1f} img/s | ETA: {remaining:.0f}s"
        )
        sys.stdout.flush()

    print(f"\n\n{'=' * 50}")
    print(f"Processing Complete!")
    print(f"Processed: {processed - skipped}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Failed: {failed}")
    print(f"Output: {output_dir}")
    print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
