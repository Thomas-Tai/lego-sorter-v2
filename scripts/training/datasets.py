"""
Dataset classes for LEGO part classification training.

Provides PyTorch Dataset implementations for:
- Legacy images (Rebrickable downloads)
- B200C synthetic images (upscaled)
- Real camera captures
- Combined/Synthetic dataset for Stage 1
"""

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def create_transforms(transform_config: list) -> A.Compose:
    """
    Create Albumentations transform pipeline from config.

    Args:
        transform_config: List of transform configurations from YAML

    Returns:
        Albumentations Compose object
    """
    if not transform_config:
        # Default minimal transform
        return A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    transforms = []
    for t_config in transform_config:
        name = t_config.get("name")
        params = {k: v for k, v in t_config.items() if k not in ["name", "transforms"]}

        if name == "ToTensorV2":
            transforms.append(ToTensorV2())
        elif name == "Normalize":
            transforms.append(A.Normalize(**params))
        elif name == "OneOf":
            # Handle nested transforms
            inner_transforms = []
            for inner in t_config.get("transforms", []):
                inner_name = inner.get("name")
                inner_params = {k: v for k, v in inner.items() if k != "name"}
                inner_cls = getattr(A, inner_name, None)
                if inner_cls:
                    inner_transforms.append(inner_cls(**inner_params))
            if inner_transforms:
                transforms.append(A.OneOf(inner_transforms, p=params.get("p", 0.5)))
        else:
            # Try to get transform class from albumentations
            transform_cls = getattr(A, name, None)
            if transform_cls:
                transforms.append(transform_cls(**params))
            else:
                logger.warning(f"Unknown transform: {name}")

    return A.Compose(transforms)


class LegacyDataset(Dataset):
    """
    Dataset for Legacy LEGO images from Rebrickable.

    Directory structure: flat directory with {part_num}_{color_id}.jpg files
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        image_size: int = 224,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size

        # Scan for images
        self.samples = []
        self.part_to_idx = {}
        self.idx_to_part = {}

        self._scan_directory(max_samples)

    def _scan_directory(self, max_samples: Optional[int] = None):
        """Scan directory for legacy images."""
        if not self.root_dir.exists():
            logger.warning(f"Legacy directory not found: {self.root_dir}")
            return

        image_files = list(self.root_dir.glob("*.jpg")) + list(
            self.root_dir.glob("*.png")
        )
        logger.info(f"Found {len(image_files)} legacy images")

        # Group by part_id (ignore color for Stage 1)
        part_images = {}
        for img_path in image_files:
            # Parse filename: {part_num}_{color_id}.jpg
            stem = img_path.stem
            parts = stem.rsplit("_", 1)
            if len(parts) >= 1:
                part_id = parts[0]
                if part_id not in part_images:
                    part_images[part_id] = []
                part_images[part_id].append(img_path)

        # Create class mapping
        for idx, part_id in enumerate(sorted(part_images.keys())):
            self.part_to_idx[part_id] = idx
            self.idx_to_part[idx] = part_id

        # Flatten to samples list
        for part_id, images in part_images.items():
            class_idx = self.part_to_idx[part_id]
            for img_path in images:
                self.samples.append((img_path, class_idx, part_id))

        # Limit samples if specified
        if max_samples and len(self.samples) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(self.samples), size=max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

        logger.info(
            f"Legacy dataset: {len(self.samples)} samples, {len(self.part_to_idx)} classes"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, class_idx, part_id = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            # Return a blank image if loading fails
            image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LANCZOS4,
            )

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, class_idx

    @property
    def num_classes(self) -> int:
        return len(self.part_to_idx)


class B200CDataset(Dataset):
    """
    Dataset for B200C LEGO images (processed/upscaled).

    Directory structure: part_id/view_idx.jpg
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        views_per_part: int = 200,
        image_size: int = 224,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        self.views_per_part = views_per_part

        # Scan for images
        self.samples = []
        self.part_to_idx = {}
        self.idx_to_part = {}

        self._scan_directory(max_samples)

    def _scan_directory(self, max_samples: Optional[int] = None):
        """Scan directory for B200C images."""
        if not self.root_dir.exists():
            logger.warning(f"B200C directory not found: {self.root_dir}")
            return

        # Get part directories
        part_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(part_dirs)} B200C parts")

        # Create class mapping
        for idx, part_dir in enumerate(sorted(part_dirs, key=lambda x: x.name)):
            part_id = part_dir.name
            self.part_to_idx[part_id] = idx
            self.idx_to_part[idx] = part_id

        # Collect samples
        for part_dir in part_dirs:
            part_id = part_dir.name
            class_idx = self.part_to_idx[part_id]

            images = list(part_dir.glob("*.jpg")) + list(part_dir.glob("*.png"))
            # Limit views per part
            images = images[: self.views_per_part]

            for img_path in images:
                self.samples.append((img_path, class_idx, part_id))

        # Limit samples if specified
        if max_samples and len(self.samples) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(self.samples), size=max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

        logger.info(
            f"B200C dataset: {len(self.samples)} samples, {len(self.part_to_idx)} classes"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, class_idx, part_id = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize if needed (B200C processed should already be 224x224)
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LANCZOS4,
            )

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, class_idx

    @property
    def num_classes(self) -> int:
        return len(self.part_to_idx)


class SyntheticDataset(Dataset):
    """
    Combined dataset for Stage 1 training (Legacy + B200C).

    Merges both datasets and creates a unified class mapping based on B200C classes
    (200 parts). Legacy images are mapped to their closest B200C class if available.
    """

    def __init__(
        self,
        legacy_dir: Path,
        b200c_dir: Path,
        transform: Optional[Callable] = None,
        legacy_config: Optional[dict] = None,
        b200c_config: Optional[dict] = None,
        split: str = "train",
        val_split: float = 0.1,
        seed: int = 42,
        image_size: int = 224,
    ):
        self.transform = transform
        self.image_size = image_size
        self.split = split

        legacy_config = legacy_config or {}
        b200c_config = b200c_config or {}

        # Storage
        self.samples = []  # List of (path, class_idx, source)
        self.part_to_idx = {}
        self.idx_to_part = {}

        # Load B200C first (defines the class space)
        b200c_samples = []
        if b200c_config.get("enabled", True) and b200c_dir.exists():
            b200c_samples = self._load_b200c(
                b200c_dir,
                max_samples=b200c_config.get("max_samples"),
                views_per_part=b200c_config.get("views_per_part", 200),
            )
            logger.info(f"Loaded {len(b200c_samples)} B200C samples")

        # Load Legacy (map to existing classes)
        legacy_samples = []
        if legacy_config.get("enabled", True) and legacy_dir.exists():
            legacy_samples = self._load_legacy(
                legacy_dir,
                max_samples=legacy_config.get("max_samples"),
            )
            logger.info(f"Loaded {len(legacy_samples)} Legacy samples")

        # Combine samples
        all_samples = b200c_samples + legacy_samples
        logger.info(f"Total samples: {len(all_samples)}")

        # Train/val split
        if len(all_samples) > 0:
            train_samples, val_samples = train_test_split(
                all_samples,
                test_size=val_split,
                random_state=seed,
                stratify=[s[1] for s in all_samples],  # Stratify by class
            )

            self.samples = train_samples if split == "train" else val_samples
            logger.info(f"{split.capitalize()} split: {len(self.samples)} samples")

    def _load_b200c(
        self,
        root_dir: Path,
        max_samples: Optional[int] = None,
        views_per_part: int = 200,
    ) -> list:
        """Load B200C samples and build class mapping."""
        samples = []

        part_dirs = sorted(
            [d for d in root_dir.iterdir() if d.is_dir()], key=lambda x: x.name
        )

        # Build class mapping from B200C parts
        for idx, part_dir in enumerate(part_dirs):
            part_id = part_dir.name
            self.part_to_idx[part_id] = idx
            self.idx_to_part[idx] = part_id

        # Collect samples
        for part_dir in part_dirs:
            part_id = part_dir.name
            class_idx = self.part_to_idx[part_id]

            images = sorted(list(part_dir.glob("*.jpg")) + list(part_dir.glob("*.png")))
            images = images[:views_per_part]

            for img_path in images:
                samples.append((img_path, class_idx, "b200c"))

        # Limit if needed
        if max_samples and len(samples) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(samples), size=max_samples, replace=False)
            samples = [samples[i] for i in indices]

        return samples

    def _load_legacy(
        self,
        root_dir: Path,
        max_samples: Optional[int] = None,
    ) -> list:
        """Load Legacy samples, mapping to existing B200C classes."""
        samples = []

        image_files = list(root_dir.glob("*.jpg")) + list(root_dir.glob("*.png"))

        for img_path in image_files:
            # Parse filename: {part_num}_{color_id}.jpg
            stem = img_path.stem
            parts = stem.rsplit("_", 1)
            if len(parts) >= 1:
                part_id = parts[0]

                # Only include if part exists in B200C class space
                if part_id in self.part_to_idx:
                    class_idx = self.part_to_idx[part_id]
                    samples.append((img_path, class_idx, "legacy"))

        # Limit if needed
        if max_samples and len(samples) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(samples), size=max_samples, replace=False)
            samples = [samples[i] for i in indices]

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, class_idx, source = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LANCZOS4,
            )

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, class_idx

    @property
    def num_classes(self) -> int:
        return len(self.part_to_idx)

    def get_class_mapping(self) -> dict:
        """Return the part_id to class_idx mapping."""
        return self.part_to_idx.copy()

    def save_class_mapping(self, output_path: Path):
        """Save class mapping to JSON file."""
        mapping = {
            "part_to_idx": self.part_to_idx,
            "idx_to_part": self.idx_to_part,
            "num_classes": self.num_classes,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(mapping, f, indent=2)


class RealCapturesDataset(Dataset):
    """
    Dataset for real camera captures (Stage 2 training).

    Directory structure: part_id/color_id/images.jpg
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[Callable] = None,
        part_mapping: Optional[dict] = None,
        color_mapping: Optional[dict] = None,
        image_size: int = 224,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size

        self.samples = []
        self.part_to_idx = part_mapping or {}
        self.color_to_idx = color_mapping or {}

        self._scan_directory()

    def _scan_directory(self):
        """Scan directory for real captures."""
        if not self.root_dir.exists():
            logger.warning(f"Real captures directory not found: {self.root_dir}")
            return

        # Build mappings if not provided
        build_part_mapping = len(self.part_to_idx) == 0
        build_color_mapping = len(self.color_to_idx) == 0

        part_idx_counter = 0
        color_idx_counter = 0

        # Scan part directories
        for part_dir in sorted(self.root_dir.iterdir()):
            if not part_dir.is_dir():
                continue

            part_id = part_dir.name

            if build_part_mapping and part_id not in self.part_to_idx:
                self.part_to_idx[part_id] = part_idx_counter
                part_idx_counter += 1

            # Scan color directories
            for color_dir in sorted(part_dir.iterdir()):
                if not color_dir.is_dir():
                    continue

                color_id = color_dir.name

                if build_color_mapping and color_id not in self.color_to_idx:
                    self.color_to_idx[color_id] = color_idx_counter
                    color_idx_counter += 1

                # Get images
                images = list(color_dir.glob("*.jpg")) + list(color_dir.glob("*.png"))

                for img_path in images:
                    if part_id in self.part_to_idx and color_id in self.color_to_idx:
                        self.samples.append(
                            (
                                img_path,
                                self.part_to_idx[part_id],
                                self.color_to_idx[color_id],
                                part_id,
                                color_id,
                            )
                        )

        logger.info(
            f"Real captures: {len(self.samples)} samples, "
            f"{len(self.part_to_idx)} parts, {len(self.color_to_idx)} colors"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, part_idx, color_idx, part_id, color_id = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LANCZOS4,
            )

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, part_idx, color_idx

    @property
    def num_parts(self) -> int:
        return len(self.part_to_idx)

    @property
    def num_colors(self) -> int:
        return len(self.color_to_idx)
