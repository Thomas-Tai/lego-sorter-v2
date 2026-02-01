#!/usr/bin/env python
"""
Dataset Validation Script
Validates captured LEGO images for AI training readiness.
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "images" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "images" / "processed"
OUTPUT_SIZE = 400

def find_turntable_circle(image):
    """Detect circular turntable in image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=200,
        maxRadius=500
    )
    
    if circles is not None:
        circles = np.round(circles).astype("int")
        largest = max(circles[0], key=lambda c: c[2])
        return tuple(map(int, largest))
    return None

def preprocess_image(image_path, output_size=400, padding=30):
    """Preprocess single image: detect turntable and crop."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None, "Failed to read"
    
    circle = find_turntable_circle(image)
    if circle is None:
        # Fallback: center crop
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        radius = min(h, w) // 3
    else:
        center_x, center_y, radius = circle
    
    # Calculate crop box with padding
    crop_size = int((radius + padding) * 2)
    half_size = crop_size // 2
    
    x1 = max(0, center_x - half_size)
    y1 = max(0, center_y - half_size)
    x2 = min(image.shape[1], center_x + half_size)
    y2 = min(image.shape[0], center_y + half_size)
    
    cropped = image[y1:y2, x1:x2]
    
    # Resize to target
    if cropped.shape[0] > 0 and cropped.shape[1] > 0:
        resized = cv2.resize(cropped, (output_size, output_size))
    else:
        return None, "Invalid crop"
    
    return resized, "OK"

def validate_dataset():
    """Main validation function."""
    # Create report file
    report_path = PROJECT_ROOT / "validation_report.txt"
    try:
        f = open(report_path, "w", encoding="utf-8")
    except Exception as e:
        print(f"Error opening report file: {e}")
        return

    def log(msg=""):
        try:
            print(msg)
            f.write(msg + "\n")
        except Exception:
            pass # Ignore encoding errors in print if any

    log("=" * 60)
    log("LEGO Dataset Validation")
    log("=" * 60)
    
    if not RAW_DIR.exists():
        log(f"ERROR: Raw directory not found: {RAW_DIR}")
        f.close()
        return
    
    # Scan raw images
    parts = defaultdict(list)
    for part_dir in RAW_DIR.iterdir():
        if not part_dir.is_dir():
            continue
        for color_dir in part_dir.iterdir():
            if not color_dir.is_dir():
                continue
            for img_file in color_dir.glob("*.jpg"):
                parts[part_dir.name].append(img_file)
    
    log(f"\nFound {len(parts)} parts with images:")
    total_images = 0
    for part_id, images in sorted(parts.items()):
        log(f"  {part_id}: {len(images)} images")
        total_images += len(images)
    
    log(f"\nTotal: {total_images} images")
    
    # Phase 1: Preprocessing test
    log("\n" + "=" * 60)
    log("Phase 1: Preprocessing Test")
    log("=" * 60)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    sample_part = None
    
    for part_id, images in parts.items():
        for img_path in images[:2]:  # Test first 2 images per part
            result, status = preprocess_image(img_path, OUTPUT_SIZE)
            if result is not None:
                success_count += 1
                if sample_part is None:
                    sample_part = (img_path, result)
            else:
                fail_count += 1
                debug_info = ""
                if status == "Invalid crop":
                     try:
                         # Re-read to debug
                         img = cv2.imread(str(img_path))
                         circ = find_turntable_circle(img)
                         if circ:
                             debug_info = f"Circle: {circ}"
                         else:
                             debug_info = "Circle: None (Fallback)"
                     except Exception as e:
                         debug_info = f"Debug Error: {e}"
                log(f"  FAIL: {img_path.name} - {status} {debug_info}")
    
    log(f"\nPreprocessing test: {success_count} OK, {fail_count} failed")
    
    # Save sample processed image
    if sample_part:
        sample_output = PROCESSED_DIR / "sample_preprocessed.jpg"
        cv2.imwrite(str(sample_output), sample_part[1])
        log(f"Sample saved: {sample_output}")
    
    # Phase 2: Check image quality
    log("\n" + "=" * 60)
    log("Phase 2: Image Quality Analysis")
    log("=" * 60)
    
    sizes = []
    brightnesses = []
    
    for part_id, images in list(parts.items())[:5]:  # First 5 parts
        for img_path in images[:4]:  # First 4 images
            img = cv2.imread(str(img_path))
            if img is not None:
                sizes.append(img.shape)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightnesses.append(np.mean(gray))
    
    if sizes:
        log(f"Image resolution: {sizes[0][1]}x{sizes[0][0]}")
        log(f"Consistent size: {len(set(sizes)) == 1}")
        log(f"Average brightness: {np.mean(brightnesses):.1f} (ideal: 100-180)")
    
    # Summary
    log("\n" + "=" * 60)
    log("Validation Summary")
    log("=" * 60)
    
    issues = []
    
    if total_images < 100:
        issues.append(f"[WARN] Only {total_images} images (recommend 500+)")
    
    if len(parts) < 10:
        issues.append(f"[WARN] Only {len(parts)} parts (recommend 20+)")
    
    if fail_count > 0:
        issues.append(f"[FAIL] {fail_count} preprocessing failures")
    
    if np.mean(brightnesses) < 80 or np.mean(brightnesses) > 200:
        issues.append("[WARN] Lighting may be too dark/bright")
    
    if issues:
        log("Issues found:")
        for issue in issues:
            log(f"  {issue}")
    else:
        log("[OK] All checks passed!")
    
    log(f"\n{'='*60}")
    log("Ready for training: " + ("[YES]" if not issues else "[REVIEW ISSUES]"))
    log("=" * 60)
    
    f.close()

if __name__ == "__main__":
    validate_dataset()
