"""
B200C Dataset Preparation Script

Prepares the B200C LEGO Classification Dataset for Stage 1 training:
1. Samples N views uniformly from each part's 360 views
2. Upscales images from 64x64 to 224x224 using Lanczos interpolation
3. Saves processed images to output directory

Usage:
    python scripts/data/prepare_b200c.py
    python scripts/data/prepare_b200c.py --config config/stage1_synthetic.yaml
    python scripts/data/prepare_b200c.py --views-per-part 200 --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import json

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def load_config(config_path: Optional[Path]) -> dict:
    """Load configuration from YAML file."""
    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def get_part_dirs(b200c_dir: Path) -> list[Path]:
    """Get list of part directories in B200C dataset."""
    part_dirs = []
    for item in b200c_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            part_dirs.append(item)
    return sorted(part_dirs)


def get_image_files(part_dir: Path) -> list[Path]:
    """Get list of image files in a part directory, sorted by view angle."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    for item in part_dir.iterdir():
        if item.suffix.lower() in image_extensions:
            images.append(item)

    # Sort by filename (assumes format like "0.jpg", "1.jpg", ..., "359.jpg")
    def sort_key(p: Path) -> int:
        try:
            return int(p.stem)
        except ValueError:
            return 0

    return sorted(images, key=sort_key)


def sample_views_uniform(images: list[Path], n_views: int) -> list[Path]:
    """Sample N views uniformly distributed across all views."""
    if n_views >= len(images):
        return images

    # Calculate indices for uniform sampling
    indices = np.linspace(0, len(images) - 1, n_views, dtype=int)
    return [images[i] for i in indices]


def sample_views_random(
    images: list[Path], n_views: int, seed: Optional[int] = None
) -> list[Path]:
    """Sample N views randomly."""
    if n_views >= len(images):
        return images

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(images), size=n_views, replace=False)
    return [images[i] for i in sorted(indices)]


def upscale_image(
    image: np.ndarray, target_size: int = 224, interpolation: int = cv2.INTER_LANCZOS4
) -> np.ndarray:
    """Upscale image to target size using specified interpolation."""
    return cv2.resize(image, (target_size, target_size), interpolation=interpolation)


def process_image(
    input_path: Path,
    output_path: Path,
    target_size: int = 224,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> bool:
    """Process a single image: load, upscale, save."""
    try:
        # Load image
        img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if img is None:
            return False

        # Upscale
        img_upscaled = upscale_image(img, target_size, interpolation)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img_upscaled)
        return True

    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")
        return False


def create_part_mapping(part_dirs: list[Path], output_path: Path) -> dict:
    """Create and save part ID to class index mapping."""
    mapping = {}
    for idx, part_dir in enumerate(part_dirs):
        part_id = part_dir.name
        mapping[part_id] = {
            "index": idx,
            "part_id": part_id,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)

    return mapping


def prepare_b200c(
    b200c_dir: Path,
    output_dir: Path,
    views_per_part: int = 200,
    target_size: int = 224,
    sampling_method: str = "uniform",
    skip_existing: bool = True,
    dry_run: bool = False,
    seed: int = 42,
) -> dict:
    """
    Prepare B200C dataset for training.

    Args:
        b200c_dir: Path to B200C dataset (containing part directories)
        output_dir: Path to output directory for processed images
        views_per_part: Number of views to sample per part
        target_size: Target image size (224 for EfficientNet)
        sampling_method: 'uniform' or 'random'
        skip_existing: Skip already processed images
        dry_run: Don't actually process, just show what would be done
        seed: Random seed for reproducibility

    Returns:
        Dictionary with processing statistics
    """
    logger = logging.getLogger(__name__)

    # Get all part directories
    part_dirs = get_part_dirs(b200c_dir)
    logger.info(f"Found {len(part_dirs)} parts in B200C dataset")

    if len(part_dirs) == 0:
        logger.error(f"No part directories found in {b200c_dir}")
        return {"error": "No parts found"}

    # Create part mapping
    mapping_path = output_dir / "part_mapping.json"
    if not dry_run:
        part_mapping = create_part_mapping(part_dirs, mapping_path)
        logger.info(f"Created part mapping with {len(part_mapping)} parts")

    # Processing statistics
    stats = {
        "total_parts": len(part_dirs),
        "views_per_part": views_per_part,
        "expected_total": len(part_dirs) * views_per_part,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
    }

    # Select interpolation method
    interpolation = cv2.INTER_LANCZOS4

    # Process each part
    for part_dir in tqdm(part_dirs, desc="Processing parts"):
        part_id = part_dir.name
        part_output_dir = output_dir / part_id

        # Get all images for this part
        images = get_image_files(part_dir)
        if len(images) == 0:
            logger.warning(f"No images found for part {part_id}")
            continue

        # Sample views
        if sampling_method == "uniform":
            sampled = sample_views_uniform(images, views_per_part)
        elif sampling_method == "random":
            sampled = sample_views_random(images, views_per_part, seed)
        else:
            sampled = images[:views_per_part]

        # Process sampled images
        for img_path in sampled:
            output_path = part_output_dir / f"{img_path.stem}.jpg"

            # Skip if exists and flag set
            if skip_existing and output_path.exists():
                stats["skipped"] += 1
                continue

            if dry_run:
                stats["processed"] += 1
                continue

            # Process image
            if process_image(img_path, output_path, target_size, interpolation):
                stats["processed"] += 1
            else:
                stats["failed"] += 1

    # Log summary
    logger.info("=" * 50)
    logger.info("B200C Processing Complete")
    logger.info(f"  Total parts: {stats['total_parts']}")
    logger.info(f"  Views per part: {stats['views_per_part']}")
    logger.info(f"  Images processed: {stats['processed']}")
    logger.info(f"  Images skipped: {stats['skipped']}")
    logger.info(f"  Images failed: {stats['failed']}")
    logger.info(f"  Output directory: {output_dir}")

    # Save stats
    if not dry_run:
        stats_path = output_dir / "processing_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare B200C dataset for Stage 1 training"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/stage1_synthetic.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--b200c-dir",
        type=Path,
        help="Override B200C source directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory",
    )
    parser.add_argument(
        "--views-per-part",
        type=int,
        default=200,
        help="Number of views to sample per part (default: 200)",
    )
    parser.add_argument(
        "--sampling",
        choices=["uniform", "random"],
        default="uniform",
        help="Sampling method (default: uniform)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=224,
        help="Target image size (default: 224)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess existing images",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without processing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    # Load config
    config = load_config(args.config)
    b200c_config = config.get("b200c_processing", {})
    data_config = config.get("data", {})

    # Determine paths (CLI args override config)
    b200c_dir = args.b200c_dir or Path(
        data_config.get(
            "b200c_dir",
            "C:/D/WorkSpace/[Local]_Station/01_Heavy_Assets/LegoSorterProject/Data/datasets/B200C LEGO Classification Dataset/64",
        )
    )
    output_dir = args.output_dir or Path(
        data_config.get(
            "b200c_processed_dir",
            "C:/D/WorkSpace/[Local]_Station/01_Heavy_Assets/LegoSorterProject/Data/images/b200c_processed",
        )
    )

    # Determine processing parameters
    views_per_part = args.views_per_part or b200c_config.get("sampling", {}).get(
        "views_per_part", 200
    )
    sampling_method = args.sampling or b200c_config.get("sampling", {}).get(
        "method", "uniform"
    )
    target_size = args.target_size or data_config.get("image_size", 224)
    skip_existing = not args.no_skip_existing and b200c_config.get(
        "skip_existing", True
    )

    logger.info("B200C Dataset Preparation")
    logger.info("=" * 50)
    logger.info(f"Source directory: {b200c_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Views per part: {views_per_part}")
    logger.info(f"Sampling method: {sampling_method}")
    logger.info(f"Target size: {target_size}x{target_size}")
    logger.info(f"Skip existing: {skip_existing}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 50)

    # Validate source directory
    if not b200c_dir.exists():
        logger.error(f"B200C directory not found: {b200c_dir}")
        sys.exit(1)

    # Run preparation
    stats = prepare_b200c(
        b200c_dir=b200c_dir,
        output_dir=output_dir,
        views_per_part=views_per_part,
        target_size=target_size,
        sampling_method=sampling_method,
        skip_existing=skip_existing,
        dry_run=args.dry_run,
        seed=args.seed,
    )

    if "error" in stats:
        sys.exit(1)


if __name__ == "__main__":
    main()
