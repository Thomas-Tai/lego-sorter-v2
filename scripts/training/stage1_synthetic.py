"""
Stage 1: Synthetic Pre-training

Train EfficientNetB0 backbone on Legacy + B200C synthetic data to learn
LEGO part shape and structure features.

Usage:
    python scripts/training/stage1_synthetic.py
    python scripts/training/stage1_synthetic.py --config config/stage1_synthetic.yaml
    python scripts/training/stage1_synthetic.py --epochs 20 --batch-size 32

Output:
    - checkpoints/stage1/backbone_synthetic.pth
    - logs/stage1/training.log
    - metrics/stage1_metrics.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.datasets import SyntheticDataset, create_transforms
from scripts.training.utils import (
    AverageMeter,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    setup_logging,
    seed_everything,
)


class Stage1Model(nn.Module):
    """
    EfficientNetB0 backbone with classification head for Stage 1 pre-training.

    The backbone learns LEGO part shape features from synthetic data.
    After training, we extract the backbone (without classifier) for Stage 2.
    """

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_layers: int = 0,
    ):
        super().__init__()

        # Load pretrained EfficientNetB0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Get feature dimension
        self.feature_dim = self.backbone.classifier[1].in_features  # 1280

        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

        # Optionally freeze early layers
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

    def _freeze_layers(self, n_blocks: int):
        """Freeze the first n blocks of the backbone."""
        for idx, (name, param) in enumerate(self.backbone.features.named_parameters()):
            block_num = int(name.split(".")[0]) if name[0].isdigit() else 0
            if block_num < n_blocks:
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone + classifier."""
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier (1280-dim)."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        return x.flatten(1)

    def get_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Extract L2-normalized embedding."""
        features = self.get_features(x)
        if normalize:
            features = nn.functional.normalize(features, dim=1)
        return features


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    gradient_clip: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        # Compute accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        acc = correct / labels.size(0)

        # Update meters
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(acc, labels.size(0))

        pbar.set_postfix(
            {"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.4f}"}
        )

    return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Validate the model."""
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    top5_meter = AverageMeter()

    pbar = tqdm(val_loader, desc="Validation", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Top-1 accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        acc = correct / labels.size(0)

        # Top-5 accuracy
        _, top5_pred = outputs.topk(5, dim=1)
        top5_correct = top5_pred.eq(labels.view(-1, 1)).any(dim=1).sum().item()
        top5_acc = top5_correct / labels.size(0)

        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(acc, labels.size(0))
        top5_meter.update(top5_acc, labels.size(0))

    return {
        "loss": loss_meter.avg,
        "accuracy": acc_meter.avg,
        "top5_accuracy": top5_meter.avg,
    }


def train_stage1(config: dict) -> dict:
    """
    Main training function for Stage 1.

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

    # Set seed for reproducibility
    seed = data_config.get("seed", 42)
    seed_everything(seed)

    # Create output directories (handle both relative and absolute paths)
    checkpoint_path = output_config.get("checkpoint_dir", "checkpoints/stage1")
    log_path = output_config.get("log_dir", "logs/stage1")

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
    logger.info("Loading datasets...")
    train_dataset = SyntheticDataset(
        legacy_dir=Path(data_config.get("legacy_dir", "")),
        b200c_dir=Path(data_config.get("b200c_processed_dir", "")),
        transform=train_transform,
        legacy_config=data_config.get("sources", {}).get("legacy", {}),
        b200c_config=data_config.get("sources", {}).get("b200c", {}),
        split="train",
        val_split=data_config.get("val_split", 0.1),
        seed=seed,
    )

    val_dataset = SyntheticDataset(
        legacy_dir=Path(data_config.get("legacy_dir", "")),
        b200c_dir=Path(data_config.get("b200c_processed_dir", "")),
        transform=val_transform,
        legacy_config=data_config.get("sources", {}).get("legacy", {}),
        b200c_config=data_config.get("sources", {}).get("b200c", {}),
        split="val",
        val_split=data_config.get("val_split", 0.1),
        seed=seed,
    )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    logger.info(f"Number of classes: {train_dataset.num_classes}")

    # Create data loaders
    batch_size = training_config.get("batch_size", 64)
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
    model = Stage1Model(
        num_classes=train_dataset.num_classes,
        pretrained=model_config.get("pretrained", "imagenet") == "imagenet",
        dropout=model_config.get("dropout", 0.3),
        freeze_layers=model_config.get("freeze_layers", 5),
    )
    model = model.to(device)
    logger.info(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.get("learning_rate", 1e-4),
        weight_decay=training_config.get("weight_decay", 0.01),
    )

    # Learning rate scheduler
    epochs = training_config.get("epochs", 15)
    scheduler_params = training_config.get("scheduler_params", {})
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=scheduler_params.get("T_max", epochs),
        eta_min=scheduler_params.get("eta_min", 1e-6),
    )

    # Mixed precision scaler
    scaler = (
        GradScaler()
        if training_config.get("mixed_precision", True) and device.type == "cuda"
        else None
    )

    # Early stopping
    early_stopping_config = training_config.get("early_stopping", {})
    early_stopping = (
        EarlyStopping(
            patience=early_stopping_config.get("patience", 5),
            min_delta=early_stopping_config.get("min_delta", 0.001),
        )
        if early_stopping_config.get("enabled", True)
        else None
    )

    # TensorBoard writer
    writer = SummaryWriter(log_dir) if output_config.get("tensorboard", True) else None

    # Training metrics
    metrics = {
        "config": config,
        "train_history": [],
        "val_history": [],
        "best_val_loss": float("inf"),
        "best_val_acc": 0.0,
        "best_epoch": 0,
    }

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting Stage 1 Training")
    logger.info("=" * 60)

    start_time = time.time()
    gradient_clip = training_config.get("gradient_clip", 1.0)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch}/{epochs}")

        # Unfreeze all layers after first few epochs
        if epoch == 3 and model_config.get("freeze_layers", 0) > 0:
            logger.info("Unfreezing all layers")
            model.unfreeze_all()

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, gradient_clip
        )
        logger.info(
            f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}"
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(
            f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val Top-5: {val_metrics['top5_accuracy']:.4f}"
        )

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"  Learning Rate: {current_lr:.2e}")

        # Record metrics
        metrics["train_history"].append(train_metrics)
        metrics["val_history"].append(val_metrics)

        # TensorBoard logging
        if writer:
            writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
            writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
            writer.add_scalar("Accuracy/val_top5", val_metrics["top5_accuracy"], epoch)
            writer.add_scalar("LR", current_lr, epoch)

        # Save best model
        if val_metrics["loss"] < metrics["best_val_loss"]:
            metrics["best_val_loss"] = val_metrics["loss"]
            metrics["best_val_acc"] = val_metrics["accuracy"]
            metrics["best_epoch"] = epoch

            best_path = checkpoint_dir / output_config.get(
                "best_model_name", "backbone_synthetic.pth"
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

        # Save periodic checkpoints
        save_every = training_config.get("checkpoint", {}).get("save_every", 5)
        if epoch % save_every == 0:
            epoch_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, epoch_path)

        # Early stopping check
        if early_stopping and early_stopping(val_metrics["loss"]):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

        epoch_time = time.time() - epoch_start
        logger.info(f"  Epoch time: {epoch_time:.1f}s")

    # Save final model
    final_path = checkpoint_dir / "backbone_synthetic_final.pth"
    save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, final_path)

    # Training complete
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  Total time: {total_time / 60:.1f} minutes")
    logger.info(f"  Best epoch: {metrics['best_epoch']}")
    logger.info(f"  Best val loss: {metrics['best_val_loss']:.4f}")
    logger.info(f"  Best val accuracy: {metrics['best_val_acc']:.4f}")
    logger.info("=" * 60)

    # Save metrics
    metrics["total_time_seconds"] = total_time
    metrics_path = metrics_dir / output_config.get(
        "metrics_file", "stage1_metrics.json"
    )
    with open(metrics_path, "w") as f:
        # Convert non-serializable items
        metrics_serializable = {k: v for k, v in metrics.items() if k != "config"}
        json.dump(metrics_serializable, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Close TensorBoard writer
    if writer:
        writer.close()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Synthetic Pre-training")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "stage1_synthetic.yaml",
        help="Path to config file",
    )
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--device", type=str, help="Override device (cuda/cpu)")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, PROJECT_ROOT / "logs" / "stage1")

    logger = logging.getLogger(__name__)
    logger.info("Stage 1: Synthetic Pre-training")
    logger.info(f"Config: {args.config}")

    # Load config
    if args.config.exists():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Apply CLI overrides
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
        metrics = train_stage1(config)
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
