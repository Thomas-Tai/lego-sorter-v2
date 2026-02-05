"""
Stage 2: Real Fine-tuning

Fine-tune the synthetic pre-trained backbone on real camera captures.
Uses multi-task learning with Part Head + Color Head.

Usage:
    python scripts/training/stage2_real.py
    python scripts/training/stage2_real.py --config config/stage2_real.yaml
    python scripts/training/stage2_real.py --epochs 50 --batch-size 16

Output:
    - checkpoints/stage2/backbone_final.pth
    - checkpoints/stage2/part_mapping.json
    - checkpoints/stage2/color_mapping.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm
import yaml
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.datasets import create_transforms
from scripts.training.utils import (
    AverageMeter,
    EarlyStopping,
    save_checkpoint,
    setup_logging,
    seed_everything,
)


class Stage2Model(nn.Module):
    """
    Multi-task model for Part + Color classification.

    Architecture:
        - EfficientNetB0 backbone (loaded from Stage 1)
        - Part classification head
        - Color classification head
    """

    def __init__(
        self,
        num_parts: int,
        num_colors: int,
        pretrained_path: Optional[Path] = None,
        dropout: float = 0.4,
    ):
        super().__init__()

        # Load EfficientNetB0 backbone
        self.backbone = models.efficientnet_b0(weights=None)
        self.feature_dim = self.backbone.classifier[1].in_features  # 1280

        # Remove original classifier
        self.backbone.classifier = nn.Identity()

        # Load pretrained weights from Stage 1
        if pretrained_path and pretrained_path.exists():
            self._load_pretrained(pretrained_path)

        # Part classification head
        self.part_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_parts),
        )

        # Color classification head
        self.color_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_colors),
        )

        self.num_parts = num_parts
        self.num_colors = num_colors

    def _load_pretrained(self, checkpoint_path: Path):
        """Load pretrained backbone from Stage 1 checkpoint."""
        logger = logging.getLogger(__name__)
        logger.info(f"Loading pretrained backbone from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Filter to backbone weights only (remove classifier)
        backbone_state = {}
        for key, value in state_dict.items():
            # Remove 'backbone.' prefix if present
            if key.startswith("backbone."):
                new_key = key[9:]  # Remove 'backbone.'
                # Skip classifier weights
                if not new_key.startswith("classifier"):
                    backbone_state[new_key] = value
            elif not key.startswith("classifier"):
                backbone_state[key] = value

        # Load backbone weights
        missing, unexpected = self.backbone.load_state_dict(
            backbone_state, strict=False
        )
        logger.info(f"Loaded pretrained backbone: {len(backbone_state)} weights")
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:5]}...")

    def freeze_backbone(self):
        """Freeze backbone for initial fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Tuple of (part_logits, color_logits)
        """
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = features.flatten(1)

        part_logits = self.part_head(features)
        color_logits = self.color_head(features)

        return part_logits, color_logits

    def get_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Extract L2-normalized embedding for inference."""
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = features.flatten(1)
        if normalize:
            features = nn.functional.normalize(features, dim=1)
        return features


class RealCapturesDataset(Dataset):
    """
    Dataset for real camera captures.

    Directory structure: part_id/color_id/images.png
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[A.Compose] = None,
        split: str = "train",
        val_split: float = 0.2,
        seed: int = 42,
        min_samples_per_part: int = 3,
        min_samples_per_color: int = 5,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.image_size = 224

        # Storage
        self.samples: List[Tuple[Path, int, int, str, str]] = []
        self.part_to_idx: Dict[str, int] = {}
        self.idx_to_part: Dict[int, str] = {}
        self.color_to_idx: Dict[str, int] = {}
        self.idx_to_color: Dict[int, str] = {}

        # Scan and build dataset
        all_samples = self._scan_directory(min_samples_per_part, min_samples_per_color)

        # Train/val split
        if len(all_samples) > 0:
            train_samples, val_samples = train_test_split(
                all_samples,
                test_size=val_split,
                random_state=seed,
                stratify=[s[1] for s in all_samples],  # Stratify by part
            )
            self.samples = train_samples if split == "train" else val_samples

        logger = logging.getLogger(__name__)
        logger.info(
            f"{split.capitalize()} dataset: {len(self.samples)} samples, "
            f"{len(self.part_to_idx)} parts, {len(self.color_to_idx)} colors"
        )

    def _scan_directory(
        self,
        min_samples_per_part: int,
        min_samples_per_color: int,
    ) -> List[Tuple[Path, int, int, str, str]]:
        """Scan directory and build class mappings."""
        logger = logging.getLogger(__name__)

        if not self.root_dir.exists():
            logger.warning(f"Real captures directory not found: {self.root_dir}")
            return []

        # First pass: count samples per part and color
        part_counts: Dict[str, int] = {}
        color_counts: Dict[str, int] = {}

        for part_dir in sorted(self.root_dir.iterdir()):
            if not part_dir.is_dir():
                continue
            part_id = part_dir.name

            for color_dir in sorted(part_dir.iterdir()):
                if not color_dir.is_dir():
                    continue
                color_id = color_dir.name

                # Count PNG files only (avoid duplicates with JPG)
                images = list(color_dir.glob("*.png"))
                count = len(images)

                part_counts[part_id] = part_counts.get(part_id, 0) + count
                color_counts[color_id] = color_counts.get(color_id, 0) + count

        # Filter parts and colors with minimum samples
        valid_parts = {p for p, c in part_counts.items() if c >= min_samples_per_part}
        valid_colors = {
            c for c, n in color_counts.items() if n >= min_samples_per_color
        }

        logger.info(f"Parts with >={min_samples_per_part} samples: {len(valid_parts)}")
        logger.info(
            f"Colors with >={min_samples_per_color} samples: {len(valid_colors)}"
        )

        # Build mappings
        for idx, part_id in enumerate(sorted(valid_parts)):
            self.part_to_idx[part_id] = idx
            self.idx_to_part[idx] = part_id

        for idx, color_id in enumerate(sorted(valid_colors)):
            self.color_to_idx[color_id] = idx
            self.idx_to_color[idx] = color_id

        # Second pass: collect samples
        samples = []
        for part_dir in sorted(self.root_dir.iterdir()):
            if not part_dir.is_dir():
                continue
            part_id = part_dir.name
            if part_id not in valid_parts:
                continue

            part_idx = self.part_to_idx[part_id]

            for color_dir in sorted(part_dir.iterdir()):
                if not color_dir.is_dir():
                    continue
                color_id = color_dir.name
                if color_id not in valid_colors:
                    continue

                color_idx = self.color_to_idx[color_id]

                for img_path in sorted(color_dir.glob("*.png")):
                    samples.append((img_path, part_idx, color_idx, part_id, color_id))

        logger.info(f"Total samples: {len(samples)}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        img_path, part_idx, color_idx, _, _ = self.samples[idx]

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

    def save_mappings(self, output_dir: Path):
        """Save class mappings to JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        part_mapping = {
            "part_to_idx": self.part_to_idx,
            "idx_to_part": {str(k): v for k, v in self.idx_to_part.items()},
            "num_parts": self.num_parts,
        }
        with open(output_dir / "part_mapping.json", "w") as f:
            json.dump(part_mapping, f, indent=2)

        color_mapping = {
            "color_to_idx": self.color_to_idx,
            "idx_to_color": {str(k): v for k, v in self.idx_to_color.items()},
            "num_colors": self.num_colors,
        }
        with open(output_dir / "color_mapping.json", "w") as f:
            json.dump(color_mapping, f, indent=2)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    part_criterion: nn.Module,
    color_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    gradient_clip: float,
    loss_weights: Dict[str, float],
) -> Dict[str, float]:
    """Train for one epoch with multi-task loss."""
    model.train()

    loss_meter = AverageMeter()
    part_loss_meter = AverageMeter()
    color_loss_meter = AverageMeter()
    part_acc_meter = AverageMeter()
    color_acc_meter = AverageMeter()

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, part_labels, color_labels in pbar:
        images = images.to(device, non_blocking=True)
        part_labels = part_labels.to(device, non_blocking=True)
        color_labels = color_labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type="cuda"):
                part_logits, color_logits = model(images)
                part_loss = part_criterion(part_logits, part_labels)
                color_loss = color_criterion(color_logits, color_labels)
                loss = (
                    loss_weights["part"] * part_loss
                    + loss_weights["color"] * color_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            part_logits, color_logits = model(images)
            part_loss = part_criterion(part_logits, part_labels)
            color_loss = color_criterion(color_logits, color_labels)
            loss = loss_weights["part"] * part_loss + loss_weights["color"] * color_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        # Compute accuracies
        _, part_pred = part_logits.max(1)
        part_correct = part_pred.eq(part_labels).sum().item()
        part_acc = part_correct / part_labels.size(0)

        _, color_pred = color_logits.max(1)
        color_correct = color_pred.eq(color_labels).sum().item()
        color_acc = color_correct / color_labels.size(0)

        # Update meters
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        part_loss_meter.update(part_loss.item(), batch_size)
        color_loss_meter.update(color_loss.item(), batch_size)
        part_acc_meter.update(part_acc, batch_size)
        color_acc_meter.update(color_acc, batch_size)

        pbar.set_postfix(
            {
                "loss": f"{loss_meter.avg:.4f}",
                "part_acc": f"{part_acc_meter.avg:.4f}",
                "color_acc": f"{color_acc_meter.avg:.4f}",
            }
        )

    return {
        "loss": loss_meter.avg,
        "part_loss": part_loss_meter.avg,
        "color_loss": color_loss_meter.avg,
        "part_acc": part_acc_meter.avg,
        "color_acc": color_acc_meter.avg,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    part_criterion: nn.Module,
    color_criterion: nn.Module,
    device: torch.device,
    loss_weights: Dict[str, float],
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()

    loss_meter = AverageMeter()
    part_loss_meter = AverageMeter()
    color_loss_meter = AverageMeter()
    part_acc_meter = AverageMeter()
    color_acc_meter = AverageMeter()
    part_top3_meter = AverageMeter()

    pbar = tqdm(val_loader, desc="Validation", leave=False)
    for images, part_labels, color_labels in pbar:
        images = images.to(device, non_blocking=True)
        part_labels = part_labels.to(device, non_blocking=True)
        color_labels = color_labels.to(device, non_blocking=True)

        part_logits, color_logits = model(images)
        part_loss = part_criterion(part_logits, part_labels)
        color_loss = color_criterion(color_logits, color_labels)
        loss = loss_weights["part"] * part_loss + loss_weights["color"] * color_loss

        # Part accuracy (Top-1)
        _, part_pred = part_logits.max(1)
        part_correct = part_pred.eq(part_labels).sum().item()
        part_acc = part_correct / part_labels.size(0)

        # Part Top-3 accuracy
        _, part_top3 = part_logits.topk(3, dim=1)
        part_top3_correct = (
            part_top3.eq(part_labels.view(-1, 1)).any(dim=1).sum().item()
        )
        part_top3_acc = part_top3_correct / part_labels.size(0)

        # Color accuracy
        _, color_pred = color_logits.max(1)
        color_correct = color_pred.eq(color_labels).sum().item()
        color_acc = color_correct / color_labels.size(0)

        # Update meters
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        part_loss_meter.update(part_loss.item(), batch_size)
        color_loss_meter.update(color_loss.item(), batch_size)
        part_acc_meter.update(part_acc, batch_size)
        color_acc_meter.update(color_acc, batch_size)
        part_top3_meter.update(part_top3_acc, batch_size)

    return {
        "loss": loss_meter.avg,
        "part_loss": part_loss_meter.avg,
        "color_loss": color_loss_meter.avg,
        "part_acc": part_acc_meter.avg,
        "part_top3_acc": part_top3_meter.avg,
        "color_acc": color_acc_meter.avg,
    }


def train_stage2(config: dict) -> dict:
    """
    Main training function for Stage 2.

    Args:
        config: Training configuration dictionary

    Returns:
        Dictionary with training results and metrics
    """
    logger = logging.getLogger(__name__)

    # Extract config sections
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    output_config = config.get("output", {})
    hardware_config = config.get("hardware", {})

    # Setup device
    device_str = hardware_config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # Set seed
    seed = data_config.get("seed", 42)
    seed_everything(seed)

    # Create output directories (handle absolute paths)
    checkpoint_path = output_config.get("checkpoint_dir", "checkpoints/stage2")
    log_path = output_config.get("log_dir", "logs/stage2")

    checkpoint_dir = (
        Path(checkpoint_path)
        if Path(checkpoint_path).is_absolute()
        else PROJECT_ROOT / checkpoint_path
    )
    log_dir = (
        Path(log_path) if Path(log_path).is_absolute() else PROJECT_ROOT / log_path
    )
    metrics_dir = (
        checkpoint_dir.parent / "metrics"
        if Path(checkpoint_path).is_absolute()
        else PROJECT_ROOT / "metrics"
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Create transforms
    train_transform = create_transforms(config.get("augmentation", {}).get("train", []))
    val_transform = create_transforms(config.get("augmentation", {}).get("val", []))

    # Create datasets
    logger.info("Loading real captures dataset...")
    real_dir = data_config.get("real_captures_dir", "data/images/raw_clean")
    real_path = (
        Path(real_dir) if Path(real_dir).is_absolute() else PROJECT_ROOT / real_dir
    )

    train_dataset = RealCapturesDataset(
        root_dir=real_path,
        transform=train_transform,
        split="train",
        val_split=data_config.get("val_split", 0.2),
        seed=seed,
        min_samples_per_part=data_config.get("min_samples_per_part", 3),
        min_samples_per_color=data_config.get("min_samples_per_color", 5),
    )

    val_dataset = RealCapturesDataset(
        root_dir=real_path,
        transform=val_transform,
        split="val",
        val_split=data_config.get("val_split", 0.2),
        seed=seed,
        min_samples_per_part=data_config.get("min_samples_per_part", 3),
        min_samples_per_color=data_config.get("min_samples_per_color", 5),
    )

    # Copy mappings from train to val
    val_dataset.part_to_idx = train_dataset.part_to_idx
    val_dataset.idx_to_part = train_dataset.idx_to_part
    val_dataset.color_to_idx = train_dataset.color_to_idx
    val_dataset.idx_to_color = train_dataset.idx_to_color

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    logger.info(f"Number of parts: {train_dataset.num_parts}")
    logger.info(f"Number of colors: {train_dataset.num_colors}")

    # Save class mappings
    if output_config.get("save_mappings", True):
        train_dataset.save_mappings(checkpoint_dir)
        logger.info(f"Saved class mappings to {checkpoint_dir}")

    # Create data loaders
    batch_size = training_config.get("batch_size", 32)
    num_workers = data_config.get("num_workers", 4)
    pin_memory = data_config.get("pin_memory", True) and device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Create model
    pretrained_path = model_config.get("pretrained_backbone")
    if pretrained_path:
        pretrained_path = (
            Path(pretrained_path)
            if Path(pretrained_path).is_absolute()
            else PROJECT_ROOT / pretrained_path
        )

    model = Stage2Model(
        num_parts=train_dataset.num_parts,
        num_colors=train_dataset.num_colors,
        pretrained_path=pretrained_path,
        dropout=model_config.get("part_head", {}).get("dropout", 0.4),
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    # Loss functions with label smoothing
    label_smoothing = training_config.get("label_smoothing", 0.1)
    part_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    color_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer with different LR for backbone
    backbone_lr_mult = training_config.get("backbone_lr_multiplier", 0.1)
    base_lr = training_config.get("learning_rate", 5e-5)

    param_groups = [
        {"params": model.backbone.parameters(), "lr": base_lr * backbone_lr_mult},
        {"params": model.part_head.parameters(), "lr": base_lr},
        {"params": model.color_head.parameters(), "lr": base_lr},
    ]

    optimizer = AdamW(
        param_groups,
        weight_decay=training_config.get("weight_decay", 0.02),
    )

    # Scheduler
    scheduler_params = training_config.get("scheduler_params", {})
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=scheduler_params.get("T_0", 10),
        T_mult=scheduler_params.get("T_mult", 1),
        eta_min=scheduler_params.get("eta_min", 1e-6),
    )

    # Mixed precision
    scaler = (
        GradScaler("cuda")
        if training_config.get("mixed_precision", True) and device.type == "cuda"
        else None
    )

    # Early stopping
    es_config = training_config.get("early_stopping", {})
    early_stopping = (
        EarlyStopping(
            patience=es_config.get("patience", 8),
            min_delta=es_config.get("min_delta", 0.001),
        )
        if es_config.get("enabled", True)
        else None
    )

    # TensorBoard
    writer = SummaryWriter(log_dir) if output_config.get("tensorboard", True) else None

    # Loss weights
    loss_weights = training_config.get("loss_weights", {"part": 1.0, "color": 0.5})

    # Training metrics
    metrics = {
        "config": config,
        "train_history": [],
        "val_history": [],
        "best_val_part_acc": 0.0,
        "best_val_color_acc": 0.0,
        "best_epoch": 0,
    }

    # Training loop
    epochs = training_config.get("epochs", 30)
    freeze_epochs = model_config.get("freeze_backbone_epochs", 2)
    gradient_clip = training_config.get("gradient_clip", 1.0)

    logger.info("=" * 60)
    logger.info("Starting Stage 2 Training")
    logger.info("=" * 60)

    # Freeze backbone initially
    if freeze_epochs > 0:
        model.freeze_backbone()
        logger.info(f"Backbone frozen for first {freeze_epochs} epochs")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch}/{epochs}")

        # Unfreeze backbone after initial epochs
        if epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            logger.info("Backbone unfrozen for full fine-tuning")

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            part_criterion,
            color_criterion,
            optimizer,
            device,
            scaler,
            gradient_clip,
            loss_weights,
        )
        logger.info(
            f"  Train - Loss: {train_metrics['loss']:.4f}, "
            f"Part Acc: {train_metrics['part_acc']:.4f}, "
            f"Color Acc: {train_metrics['color_acc']:.4f}"
        )

        # Validate
        val_metrics = validate(
            model, val_loader, part_criterion, color_criterion, device, loss_weights
        )
        logger.info(
            f"  Val   - Loss: {val_metrics['loss']:.4f}, "
            f"Part Acc: {val_metrics['part_acc']:.4f} (Top-3: {val_metrics['part_top3_acc']:.4f}), "
            f"Color Acc: {val_metrics['color_acc']:.4f}"
        )

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[1]["lr"]  # Head LR
        logger.info(f"  Learning Rate: {current_lr:.2e}")

        # Record metrics
        metrics["train_history"].append(train_metrics)
        metrics["val_history"].append(val_metrics)

        # TensorBoard
        if writer:
            writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/train_part", train_metrics["part_acc"], epoch)
            writer.add_scalar("Accuracy/val_part", val_metrics["part_acc"], epoch)
            writer.add_scalar(
                "Accuracy/val_part_top3", val_metrics["part_top3_acc"], epoch
            )
            writer.add_scalar("Accuracy/train_color", train_metrics["color_acc"], epoch)
            writer.add_scalar("Accuracy/val_color", val_metrics["color_acc"], epoch)
            writer.add_scalar("LR", current_lr, epoch)

        # Save best model (based on part accuracy)
        if val_metrics["part_acc"] > metrics["best_val_part_acc"]:
            metrics["best_val_part_acc"] = val_metrics["part_acc"]
            metrics["best_val_color_acc"] = val_metrics["color_acc"]
            metrics["best_epoch"] = epoch

            best_path = checkpoint_dir / output_config.get(
                "best_model_name", "backbone_final.pth"
            )
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_metrics,
                best_path,
            )
            logger.info(f"  Saved best model to {best_path}")

        # Periodic checkpoint
        save_every = training_config.get("checkpoint", {}).get("save_every", 10)
        if epoch % save_every == 0:
            epoch_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, epoch_path)

        # Early stopping (monitor part accuracy - want to maximize)
        if early_stopping:
            # Use negative accuracy since EarlyStopping monitors for decrease
            if early_stopping(-val_metrics["part_acc"]):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        epoch_time = time.time() - epoch_start
        logger.info(f"  Epoch time: {epoch_time:.1f}s")

    # Save final model
    final_path = checkpoint_dir / "backbone_final_last.pth"
    save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, final_path)

    # Training complete
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  Total time: {total_time / 60:.1f} minutes")
    logger.info(f"  Best epoch: {metrics['best_epoch']}")
    logger.info(f"  Best part accuracy: {metrics['best_val_part_acc']:.4f}")
    logger.info(f"  Best color accuracy: {metrics['best_val_color_acc']:.4f}")
    logger.info("=" * 60)

    # Save metrics
    metrics["total_time_seconds"] = total_time
    metrics_path = metrics_dir / output_config.get(
        "metrics_file", "stage2_metrics.json"
    )
    with open(metrics_path, "w") as f:
        metrics_serializable = {k: v for k, v in metrics.items() if k != "config"}
        json.dump(metrics_serializable, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    if writer:
        writer.close()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Real Fine-tuning")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "stage2_real.yaml",
        help="Path to config file",
    )
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--device", type=str, help="Override device")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, PROJECT_ROOT / "logs" / "stage2")

    logger = logging.getLogger(__name__)
    logger.info("Stage 2: Real Fine-tuning")
    logger.info(f"Config: {args.config}")

    # Load config
    if args.config.exists():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # CLI overrides
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.device:
        config["hardware"]["device"] = args.device

    # Train
    try:
        metrics = train_stage2(config)
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
