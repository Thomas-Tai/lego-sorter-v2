"""
Training utilities for LEGO part classification.

Provides helper classes and functions for:
- Logging setup
- Checkpoint management
- Training metrics
- Early stopping
- Reproducibility
"""

import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        log_file: Log filename (default: training_{timestamp}.log)

    Returns:
        Root logger
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"training_{timestamp}.log"

        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking loss and accuracy during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss, 'max' for accuracy
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        """
        Check if training should stop.

        Args:
            metric: Current validation metric

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = metric
            return False

        if self.mode == "min":
            improved = metric < self.best_score - self.min_delta
        else:
            improved = metric > self.best_score + self.min_delta

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            return True

        return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    epoch: int,
    metrics: dict,
    path: Path,
    extra: Optional[dict] = None,
):
    """
    Save training checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer state
        scheduler: LR scheduler state
        epoch: Current epoch
        metrics: Validation metrics
        path: Path to save checkpoint
        extra: Extra data to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load training checkpoint.

    Args:
        path: Path to checkpoint
        model: PyTorch model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: LR scheduler to load state into (optional)
        device: Device to load tensors to

    Returns:
        Checkpoint dictionary with epoch, metrics, etc.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count number of parameters in model.

    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


class GradientAccumulator:
    """
    Gradient accumulation for effective larger batch sizes.

    Useful when GPU memory is limited but larger batch sizes are needed.
    """

    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def should_step(self) -> bool:
        """Check if optimizer should step."""
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True
        return False

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient accumulation."""
        return loss / self.accumulation_steps


def compute_class_weights(
    dataset,
    num_classes: int,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    Args:
        dataset: Dataset with (image, label) tuples
        num_classes: Number of classes
        smoothing: Label smoothing factor

    Returns:
        Tensor of class weights
    """
    class_counts = np.zeros(num_classes)

    for _, label in dataset:
        if isinstance(label, (list, tuple)):
            label = label[0]  # Multi-task: use first label
        class_counts[label] += 1

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)

    # Inverse frequency weighting
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes

    # Apply smoothing
    weights = (1 - smoothing) * weights + smoothing

    return torch.FloatTensor(weights)


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> tuple:
    """
    Apply mixup augmentation.

    Args:
        x: Input images
        y: Labels
        alpha: Mixup alpha parameter

    Returns:
        Mixed images, labels_a, labels_b, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
