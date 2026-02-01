"""
Image Preprocessing Module
Provides tools for cropping and standardizing captured LEGO part images.
"""
import os
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def find_turntable_circle(image: np.ndarray) -> tuple:
    """Detect the gray circular turntable in the image.
    
    Returns:
        tuple: (center_x, center_y, radius) or None if not found
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=int(min(image.shape[:2]) * 0.25),  # At least 25% of image
        maxRadius=int(min(image.shape[:2]) * 0.5)     # At most 50% of image
    )
    
    if circles is not None:
        # Take the largest circle found
        circles = np.uint16(np.around(circles))
        largest = max(circles[0], key=lambda c: c[2])
        return int(largest[0]), int(largest[1]), int(largest[2])
    
    return None


def crop_to_square(image: np.ndarray, center_x: int, center_y: int, 
                   size: int, padding: int = 30) -> np.ndarray:
    """Crop image to a square centered on the given point.
    
    Args:
        image: Input image
        center_x, center_y: Center point of crop
        size: Size of the square (will be size x size)
        padding: Extra padding to add around edges (default 30 for long parts)
    
    Returns:
        Cropped square image
    """
    half_size = size // 2 + padding
    
    # Calculate crop bounds with boundary checking
    h, w = image.shape[:2]
    x1 = max(0, center_x - half_size)
    y1 = max(0, center_y - half_size)
    x2 = min(w, center_x + half_size)
    y2 = min(h, center_y + half_size)
    
    cropped = image[y1:y2, x1:x2]
    return cropped


def preprocess_image(input_path: str, output_path: str, 
                     target_size: int = 400) -> bool:
    """Preprocess a single image: detect turntable and crop.
    
    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        target_size: Output image size (square)
    
    Returns:
        True if successful, False otherwise
    """
    # Read image
    image = cv2.imread(input_path)
    if image is None:
        logger.warning(f"Could not read image: {input_path}")
        return False
    
    # Find turntable circle
    circle = find_turntable_circle(image)
    
    if circle is None:
        # Fallback: crop from center
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        crop_size = min(h, w) - 40
        logger.info(f"No circle found, using center crop: {input_path}")
    else:
        center_x, center_y, radius = circle
        crop_size = radius * 2
        logger.debug(f"Found turntable at ({center_x}, {center_y}), r={radius}")
    
    # Crop to square
    cropped = crop_to_square(image, center_x, center_y, crop_size)
    
    # Resize to target size
    resized = cv2.resize(cropped, (target_size, target_size), 
                         interpolation=cv2.INTER_AREA)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    cv2.imwrite(output_path, resized)
    return True


def preprocess_dataset(raw_dir: str, processed_dir: str, 
                       target_size: int = 400) -> dict:
    """Process all images in the raw directory structure.
    
    Args:
        raw_dir: Path to raw images (data/images/raw/)
        processed_dir: Path to save processed images (data/images/processed/)
        target_size: Output image size
    
    Returns:
        dict with 'success' and 'failed' counts
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    
    # Find all jpg files
    image_files = list(raw_path.glob('**/*.jpg'))
    total = len(image_files)
    
    print(f"Processing {total} images...")
    
    for i, img_path in enumerate(image_files):
        # Preserve directory structure
        relative_path = img_path.relative_to(raw_path)
        output_path = processed_path / relative_path
        
        # Skip if already processed
        if output_path.exists():
            stats['skipped'] += 1
            continue
        
        if preprocess_image(str(img_path), str(output_path), target_size):
            stats['success'] += 1
        else:
            stats['failed'] += 1
        
        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] Processed")
    
    print(f"\nComplete: {stats['success']} processed, "
          f"{stats['skipped']} skipped, {stats['failed']} failed")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess LEGO part images")
    parser.add_argument("--raw", default="data/images/raw",
                        help="Path to raw images")
    parser.add_argument("--output", default="data/images/processed",
                        help="Path for processed images")
    parser.add_argument("--size", type=int, default=400,
                        help="Output image size (square)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    preprocess_dataset(args.raw, args.output, args.size)
