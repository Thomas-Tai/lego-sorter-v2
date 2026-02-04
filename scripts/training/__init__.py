"""
Training scripts for LEGO part classification.

Stage 1: Synthetic pre-training (Legacy + B200C)
Stage 2: Real fine-tuning (camera captures)
Stage 3: Deployment (classifier or vector space)
"""

from scripts.training.datasets import (
    create_transforms,
    LegacyDataset,
    B200CDataset,
    SyntheticDataset,
    RealCapturesDataset,
)
from scripts.training.utils import (
    AverageMeter,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    setup_logging,
    seed_everything,
    count_parameters,
    get_lr,
    compute_class_weights,
    mixup_data,
    mixup_criterion,
)

__all__ = [
    # Datasets
    "create_transforms",
    "LegacyDataset",
    "B200CDataset",
    "SyntheticDataset",
    "RealCapturesDataset",
    # Utils
    "AverageMeter",
    "EarlyStopping",
    "save_checkpoint",
    "load_checkpoint",
    "setup_logging",
    "seed_everything",
    "count_parameters",
    "get_lr",
    "compute_class_weights",
    "mixup_data",
    "mixup_criterion",
]
