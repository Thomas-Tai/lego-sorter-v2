"""
Script to test hybrid background removal (Rembg + Color Filtering).
"""

import sys
import os
import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image
import io


def process_hybrid(input_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing: {input_path}")

    # 1. Load Image
    with open(input_path, "rb") as f:
        img_bytes = f.read()

    # 2. Run Rembg (Step 1)
    print("Running Rembg...")
    session = new_session("u2net")
    output_bytes = remove(img_bytes, session=session, alpha_matting=True)

    # Convert to OpenCV format (RGBA)
    img_pil = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    img_np = np.array(img_pil)

    # Save Rembg only output
    img_pil.save(os.path.join(output_dir, "step1_rembg.png"))

    # 3. Analyze Colors (Step 2 - Color Filter)
    # We want to remove "Grey" pixels that are opaque in the alpha channel
    # Convert RGB to HSV
    rgb_img = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    # DEBUG: Print HSV at known turntable points
    # From analyze_colors: (319, 179) was H=140 S=6 V=131
    # Check what hsv_img has there.
    pts = [(319, 179), (959, 179), (639, 359)]
    for x, y in pts:
        if y < hsv_img.shape[0] and x < hsv_img.shape[1]:
            p = hsv_img[y, x]
            print(f"HSV at ({x},{y}): H={p[0]} S={p[1]} V={p[2]} Alpha={img_np[y,x,3]}")

    # Define Grey Range
    # Grey is low saturation. The Value (brightness) can vary but turntable is usually mid-grey.
    # Open ranges for tuning:
    # Scheme A: Strict Grey (Sat < 20, Val > 50)
    lower_grey_a = np.array([0, 0, 50])
    upper_grey_a = np.array(
        [180, 20, 220]
    )  # Limit upper value to avoid removing white highlights if any, though turntable is darker.

    # Scheme B: Board Grey (Sat < 40, Val 50-200)
    lower_grey_b = np.array([0, 0, 50])
    upper_grey_b = np.array([180, 50, 220])

    # Data-Driven Range from analyze_colors.py
    # Turntable: S ~ 6-8, V ~ 125-131
    # Green Part: S ~ 124, V ~ 195 (Safely preserved)

    masks = [("tuned_turntable", np.array([0, 0, 100]), np.array([180, 25, 160]))]

    for name, low, high in masks:
        print(f"Applying filter: {name}")
        print(f"  Range: V=[{low[2]}-{high[2]}], S=[{low[1]}-{high[1]}]")

        mask = cv2.inRange(hsv_img, low, high)

        # Calculate impact
        pixels_removed = np.count_nonzero(mask)
        total_pixels = mask.size
        percent = (pixels_removed / total_pixels) * 100
        print(f"  Removing {pixels_removed} pixels ({percent:.2f}%)")

        result = img_np.copy()
        result[mask == 255, 3] = 0  # Apply color mask

        # Step 3: Keep Largest Component (Remove Island Noise)
        # Extract alpha channel
        alpha = result[:, :, 3]

        # Threshold alpha to get binary mask of "visible pixels"
        _, binary = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Find largest contour (The Lego)
            largest_contour = max(contours, key=cv2.contourArea)

            # Create a mask of ONLY the largest contour
            clean_mask = np.zeros_like(binary)
            cv2.drawContours(
                clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED
            )

            # Apply Clean Mask: Anything NOT in the largest contour becomes transparent
            result[clean_mask == 0, 3] = 0

            print(
                f"  Kept largest component. Removed {len(contours) - 1} noise islands."
            )

        # Save
        res_pil = Image.fromarray(result)
        res_pil.save(os.path.join(output_dir, f"step3_{name}_cleaned.png"))
        print(f"  Saved cleaned result.")

        # Save the mask itself for debug
        cv2.imwrite(os.path.join(output_dir, f"debug_{name}_mask.png"), mask)

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_hybrid.py <img_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    out_dir = os.path.join(os.path.dirname(img_path), "hybrid_test")
    process_hybrid(img_path, out_dir)
