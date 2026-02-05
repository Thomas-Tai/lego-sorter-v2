"""
Evaluate trained model and generate confusion analysis.

This script evaluates the Stage 2 model on the validation set and identifies:
- Overall accuracy metrics
- Most confused part pairs
- Parts with lowest accuracy
- Recommendations for improvement

Usage:
    python scripts/training/evaluate_model.py
    python scripts/training/evaluate_model.py --checkpoint checkpoints/stage2/backbone_final.pth
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Stage2Model(nn.Module):
    """Stage 2 model for evaluation."""

    def __init__(self, num_parts: int, num_colors: int, dropout: float = 0.4):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.part_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_parts),
        )

        self.color_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_colors),
        )

    def forward(self, x):
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = features.flatten(1)
        return self.part_head(features), self.color_head(features)


def load_model(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Infer dimensions
    num_parts = state_dict["part_head.4.weight"].shape[0]
    num_colors = state_dict["color_head.4.weight"].shape[0]

    model = Stage2Model(num_parts, num_colors)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model.to(device), num_parts, num_colors


def load_mappings(checkpoint_dir: Path):
    """Load class mappings."""
    with open(checkpoint_dir / "part_mapping.json") as f:
        part_data = json.load(f)
    with open(checkpoint_dir / "color_mapping.json") as f:
        color_data = json.load(f)

    idx_to_part = {int(k): v for k, v in part_data["idx_to_part"].items()}
    idx_to_color = {int(k): v for k, v in color_data["idx_to_color"].items()}
    part_to_idx = part_data["part_to_idx"]

    return idx_to_part, idx_to_color, part_to_idx


def get_validation_samples(data_dir: Path, part_to_idx: Dict, seed: int = 42):
    """Get validation samples (20% split, same as training)."""
    from sklearn.model_selection import train_test_split

    all_samples = []
    for part_dir in sorted(data_dir.iterdir()):
        if not part_dir.is_dir():
            continue
        part_id = part_dir.name
        if part_id not in part_to_idx:
            continue

        part_idx = part_to_idx[part_id]
        for color_dir in sorted(part_dir.iterdir()):
            if not color_dir.is_dir():
                continue
            for img_path in sorted(color_dir.glob("*.png")):
                all_samples.append((img_path, part_idx, part_id))

    if not all_samples:
        return []

    _, val_samples = train_test_split(
        all_samples,
        test_size=0.2,
        random_state=seed,
        stratify=[s[1] for s in all_samples],
    )
    return val_samples


def preprocess_image(img_path: Path, size: int = 224) -> torch.Tensor:
    """Preprocess image for model."""
    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)

    # Normalize
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # To tensor (CHW)
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).float()


def evaluate(
    model: nn.Module,
    samples: List[Tuple],
    idx_to_part: Dict,
    device: torch.device,
) -> Dict:
    """Evaluate model on samples."""
    model.eval()

    predictions = []
    confusion_pairs = defaultdict(int)  # (true, pred) -> count
    part_stats = defaultdict(lambda: {"correct": 0, "total": 0, "top3_correct": 0})

    with torch.no_grad():
        for img_path, true_idx, true_part in tqdm(samples, desc="Evaluating"):
            img_tensor = preprocess_image(img_path)
            if img_tensor is None:
                continue

            img_tensor = img_tensor.unsqueeze(0).to(device)
            part_logits, _ = model(img_tensor)

            probs = torch.softmax(part_logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            top3_idx = probs.topk(3).indices.tolist()
            confidence = probs[pred_idx].item()

            pred_part = idx_to_part.get(pred_idx, str(pred_idx))
            correct = pred_idx == true_idx
            top3_correct = true_idx in top3_idx

            predictions.append(
                {
                    "path": str(img_path),
                    "true_part": true_part,
                    "pred_part": pred_part,
                    "confidence": confidence,
                    "correct": correct,
                    "top3_correct": top3_correct,
                }
            )

            part_stats[true_part]["total"] += 1
            if correct:
                part_stats[true_part]["correct"] += 1
            if top3_correct:
                part_stats[true_part]["top3_correct"] += 1

            if not correct:
                confusion_pairs[(true_part, pred_part)] += 1

    # Calculate metrics
    total = len(predictions)
    correct = sum(1 for p in predictions if p["correct"])
    top3_correct = sum(1 for p in predictions if p["top3_correct"])

    accuracy = correct / total if total > 0 else 0
    top3_accuracy = top3_correct / total if total > 0 else 0

    # Sort confusion pairs by count
    sorted_confusion = sorted(confusion_pairs.items(), key=lambda x: -x[1])

    # Calculate per-part accuracy
    part_accuracy = {}
    for part, stats in part_stats.items():
        if stats["total"] > 0:
            part_accuracy[part] = {
                "accuracy": stats["correct"] / stats["total"],
                "top3_accuracy": stats["top3_correct"] / stats["total"],
                "total": stats["total"],
                "correct": stats["correct"],
            }

    # Sort parts by accuracy (ascending - worst first)
    worst_parts = sorted(part_accuracy.items(), key=lambda x: x[1]["accuracy"])

    return {
        "total_samples": total,
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "confusion_pairs": sorted_confusion[:20],  # Top 20 confusions
        "worst_parts": worst_parts[:15],  # 15 worst parts
        "part_accuracy": part_accuracy,
    }


def generate_recommendations(results: Dict) -> List[str]:
    """Generate improvement recommendations based on results."""
    recommendations = []

    # Check overall accuracy
    if results["accuracy"] < 0.90:
        recommendations.append(
            f"Overall accuracy ({results['accuracy']:.1%}) is below 90%. "
            "Consider: more training epochs, learning rate tuning, or more data."
        )

    # Analyze confusion pairs
    if results["confusion_pairs"]:
        top_confusions = results["confusion_pairs"][:5]
        confused_parts = set()
        for (true_part, pred_part), count in top_confusions:
            confused_parts.add(true_part)
            confused_parts.add(pred_part)

        if confused_parts:
            recommendations.append(
                f"Most confused parts: {', '.join(sorted(confused_parts))}. "
                "Consider: capturing more images of these parts from different angles."
            )

    # Analyze worst performing parts
    if results["worst_parts"]:
        very_bad = [(p, s) for p, s in results["worst_parts"] if s["accuracy"] < 0.5]
        if very_bad:
            bad_parts = [p for p, _ in very_bad[:5]]
            recommendations.append(
                f"Parts with <50% accuracy: {', '.join(bad_parts)}. "
                "Priority: add more training images for these parts."
            )

    # Check if Top-3 is much better than Top-1
    gap = results["top3_accuracy"] - results["accuracy"]
    if gap > 0.15:
        recommendations.append(
            f"Large Top-1 to Top-3 gap ({gap:.1%}). "
            "Consider: using Top-3 predictions with confidence thresholding in production."
        )

    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "stage2" / "backbone_final.pth",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "C:/D/WorkSpace/[Local]_Station/01_Heavy_Assets/LegoSorterProject/Data/images/raw_clean"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Model Evaluation")
    logger.info("=" * 60)

    # Force CPU to avoid GPU compatibility issues
    device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Load model and mappings
    logger.info(f"Loading model from {args.checkpoint}")
    model, num_parts, num_colors = load_model(args.checkpoint, device)
    logger.info(f"Model: {num_parts} parts, {num_colors} colors")

    checkpoint_dir = args.checkpoint.parent
    idx_to_part, idx_to_color, part_to_idx = load_mappings(checkpoint_dir)

    # Get validation samples
    logger.info(f"Loading validation samples from {args.data_dir}")
    samples = get_validation_samples(args.data_dir, part_to_idx)
    logger.info(f"Validation samples: {len(samples)}")

    if not samples:
        logger.error("No validation samples found!")
        return

    # Evaluate
    logger.info("\nEvaluating...")
    results = evaluate(model, samples, idx_to_part, device)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Samples: {results['total_samples']}")
    logger.info(f"Top-1 Accuracy: {results['accuracy']:.2%}")
    logger.info(f"Top-3 Accuracy: {results['top3_accuracy']:.2%}")

    logger.info("\n--- Top Confusion Pairs ---")
    for (true_part, pred_part), count in results["confusion_pairs"][:10]:
        logger.info(f"  {true_part} → {pred_part}: {count} errors")

    logger.info("\n--- Worst Performing Parts ---")
    for part, stats in results["worst_parts"][:10]:
        logger.info(
            f"  {part}: {stats['accuracy']:.1%} "
            f"({stats['correct']}/{stats['total']}), "
            f"Top-3: {stats['top3_accuracy']:.1%}"
        )

    # Generate recommendations
    recommendations = generate_recommendations(results)
    logger.info("\n--- Recommendations ---")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"  {i}. {rec}")

    # Save results
    output_path = args.output or (checkpoint_dir / "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "accuracy": results["accuracy"],
                "top3_accuracy": results["top3_accuracy"],
                "total_samples": results["total_samples"],
                "confusion_pairs": [
                    (list(k), v) for k, v in results["confusion_pairs"]
                ],
                "worst_parts": [(p, s) for p, s in results["worst_parts"]],
                "recommendations": recommendations,
            },
            f,
            indent=2,
        )
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
