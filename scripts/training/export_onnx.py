"""
Stage 3: Export trained model to ONNX for deployment.

Exports the Stage 2 multi-task classifier to ONNX format for efficient
inference on Raspberry Pi and laptop.

Usage:
    python scripts/training/export_onnx.py
    python scripts/training/export_onnx.py --checkpoint checkpoints/stage2/backbone_final.pth
    python scripts/training/export_onnx.py --output models/lego_classifier.onnx

Output:
    - models/lego_classifier.onnx (ONNX model)
    - models/part_mapping.json (class mappings - copied)
    - models/color_mapping.json (color mappings - copied)
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Stage2ModelExport(nn.Module):
    """
    Stage 2 Model for ONNX export.

    Identical architecture to Stage2Model but with separate outputs for export.
    Returns a dictionary with 'part_logits' and 'color_logits' keys.
    """

    def __init__(
        self,
        num_parts: int,
        num_colors: int,
        dropout: float = 0.4,
    ):
        super().__init__()

        # EfficientNetB0 backbone
        self.backbone = models.efficientnet_b0(weights=None)
        self.feature_dim = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Identity()

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

    def forward(self, x: torch.Tensor):
        """Forward pass returning part and color logits."""
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = features.flatten(1)

        part_logits = self.part_head(features)
        color_logits = self.color_head(features)

        return part_logits, color_logits


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load checkpoint and extract model configuration."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Infer num_parts and num_colors from weight shapes
    num_parts = None
    num_colors = None

    for key, tensor in state_dict.items():
        if "part_head.4.weight" in key:
            num_parts = tensor.shape[0]
        elif "color_head.4.weight" in key:
            num_colors = tensor.shape[0]

    if num_parts is None or num_colors is None:
        raise ValueError(
            "Could not infer model dimensions from checkpoint. "
            "Expected 'part_head.4.weight' and 'color_head.4.weight' keys."
        )

    logger.info(f"Detected {num_parts} parts, {num_colors} colors")

    return state_dict, num_parts, num_colors


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset_version: int = 14,
    simplify: bool = True,
) -> dict:
    """
    Export PyTorch checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to Stage 2 checkpoint (.pth)
        output_path: Output ONNX file path
        opset_version: ONNX opset version (default: 14)
        simplify: Whether to simplify the ONNX model (default: True)

    Returns:
        Dictionary with export metadata
    """
    logger = logging.getLogger(__name__)

    device = torch.device("cpu")  # Export on CPU for compatibility

    # Load checkpoint
    state_dict, num_parts, num_colors = load_checkpoint(checkpoint_path, device)

    # Create model
    model = Stage2ModelExport(
        num_parts=num_parts,
        num_colors=num_colors,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    logger.info(f"Model created: {num_parts} parts, {num_colors} colors")

    # Dummy input for tracing (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    logger.info(f"Exporting to ONNX: {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["part_logits", "color_logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "part_logits": {0: "batch_size"},
            "color_logits": {0: "batch_size"},
        },
    )

    logger.info("ONNX export complete")

    # Verify the export
    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed")

        # Simplify if requested
        if simplify:
            try:
                import onnxsim

                logger.info("Simplifying ONNX model...")
                simplified_model, check = onnxsim.simplify(onnx_model)
                if check:
                    onnx.save(simplified_model, str(output_path))
                    logger.info("ONNX model simplified successfully")
                else:
                    logger.warning("ONNX simplification check failed, keeping original")
            except ImportError:
                logger.warning("onnx-simplifier not installed, skipping simplification")

    except ImportError:
        logger.warning("onnx package not installed, skipping verification")

    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX model size: {file_size_mb:.2f} MB")

    # Test inference with ONNX Runtime
    try:
        import onnxruntime as ort

        logger.info("Testing ONNX inference...")

        session = ort.InferenceSession(str(output_path))
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]

        # Run test inference
        test_input = dummy_input.numpy()
        outputs = session.run(output_names, {input_name: test_input})

        logger.info(f"  Input shape: {test_input.shape}")
        logger.info(f"  Part logits shape: {outputs[0].shape}")
        logger.info(f"  Color logits shape: {outputs[1].shape}")
        logger.info("ONNX inference test passed")

    except ImportError:
        logger.warning("onnxruntime not installed, skipping inference test")

    return {
        "output_path": str(output_path),
        "num_parts": num_parts,
        "num_colors": num_colors,
        "opset_version": opset_version,
        "file_size_mb": file_size_mb,
    }


def copy_mappings(checkpoint_dir: Path, output_dir: Path):
    """Copy class mapping files to output directory."""
    logger = logging.getLogger(__name__)

    mapping_files = ["part_mapping.json", "color_mapping.json"]

    for filename in mapping_files:
        src = checkpoint_dir / filename
        dst = output_dir / filename

        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"Copied {filename} to {output_dir}")
        else:
            logger.warning(f"Mapping file not found: {src}")


def main():
    parser = argparse.ArgumentParser(description="Export Stage 2 model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "stage2" / "backbone_final.pth",
        help="Path to Stage 2 checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "lego_classifier.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip ONNX model simplification",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Stage 3: Export to ONNX")
    logger.info("=" * 60)

    # Validate checkpoint
    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.info("Expected file: checkpoints/stage2/backbone_final.pth")
        logger.info("Please run Stage 2 training first or provide correct path.")
        sys.exit(1)

    # Export
    metadata = export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        simplify=not args.no_simplify,
    )

    # Copy mapping files
    checkpoint_dir = args.checkpoint.parent
    output_dir = args.output.parent
    copy_mappings(checkpoint_dir, output_dir)

    # Summary
    logger.info("=" * 60)
    logger.info("Export Complete!")
    logger.info(f"  ONNX Model: {metadata['output_path']}")
    logger.info(f"  Model Size: {metadata['file_size_mb']:.2f} MB")
    logger.info(f"  Parts: {metadata['num_parts']}")
    logger.info(f"  Colors: {metadata['num_colors']}")
    logger.info("=" * 60)

    # Print deployment instructions
    print("\n" + "=" * 60)
    print("Deployment Instructions:")
    print("=" * 60)
    print("1. Copy these files to your deployment target:")
    print(f"   - {args.output}")
    print(f"   - {output_dir / 'part_mapping.json'}")
    print(f"   - {output_dir / 'color_mapping.json'}")
    print()
    print("2. Install ONNX Runtime on target:")
    print("   pip install onnxruntime  # CPU")
    print("   pip install onnxruntime-gpu  # GPU (optional)")
    print()
    print("3. Usage example:")
    print("   import onnxruntime as ort")
    print("   session = ort.InferenceSession('lego_classifier.onnx')")
    print("   outputs = session.run(None, {'input': image_batch})")
    print("   part_logits, color_logits = outputs")
    print("=" * 60)


if __name__ == "__main__":
    main()
