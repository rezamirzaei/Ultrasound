"""
Ultrasound Imaging Toolkit demo application.

This module provides CLI-driven demonstrations for preprocessing,
optimization, segmentation, classification, and end-to-end pipeline usage.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultrasound.data import BUSIDataset, _generate_synthetic_ultrasound, create_sample_data
from ultrasound.models.classifier import ResNetClassifier, UltrasoundClassifier
from ultrasound.models.unet import AttentionUNet, UNet, combined_loss, dice_loss
from ultrasound.preprocessing.denoising import admm_tv_denoising
from ultrasound.preprocessing.enhancement import ContrastEnhancer
from ultrasound.preprocessing.speckle import SpeckleReducer
from ultrasound.utils.visualization import plot_preprocessing_comparison, plot_speckle_analysis


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def demo_preprocessing(image: np.ndarray, output_dir: Path) -> dict[str, np.ndarray]:
    """Demonstrate preprocessing techniques for ultrasound images."""
    _ensure_output_dir(output_dir)
    print("\n" + "=" * 60)
    print("PREPROCESSING DEMONSTRATION")
    print("=" * 60)

    gray = np.mean(image, axis=2).astype(np.uint8) if image.ndim == 3 else image

    print("\n1. Speckle Reduction Techniques:")
    print("-" * 40)

    reducer_lee = SpeckleReducer(method="lee", window_size=7)
    lee_result = reducer_lee.reduce(gray)
    mean_val, cv = reducer_lee.estimate_speckle_level(gray)
    print(f"   Original - Mean: {mean_val:.2f}, CV: {cv:.3f}")
    mean_val, cv = reducer_lee.estimate_speckle_level(lee_result)
    print(f"   Lee Filter - Mean: {mean_val:.2f}, CV: {cv:.3f}")

    reducer_frost = SpeckleReducer(method="frost", window_size=5, damping_factor=1.5)
    frost_result = reducer_frost.reduce(gray)

    reducer_median = SpeckleReducer(method="median", window_size=5)
    _ = reducer_median.reduce(gray)

    print("\n2. Contrast Enhancement:")
    print("-" * 40)

    enhancer = ContrastEnhancer(method="clahe", clip_limit=2.5)
    clahe_result = enhancer.enhance(gray)

    original_stats = enhancer.analyze_contrast(gray)
    enhanced_stats = enhancer.analyze_contrast(clahe_result)
    print(f"   Original contrast ratio: {original_stats['contrast_ratio']:.3f}")
    print(f"   CLAHE contrast ratio: {enhanced_stats['contrast_ratio']:.3f}")

    print("\n3. ADMM-based Total Variation Denoising:")
    print("-" * 40)

    tv_result, convergence = admm_tv_denoising(
        gray,
        lambda_tv=0.05,
        rho=1.0,
        n_iter=30,
        verbose=False,
    )
    print(f"   Converged in {len(convergence['primal_residuals'])} iterations")
    print(f"   Final primal residual: {convergence['primal_residuals'][-1]:.6f}")

    processed = {
        "Lee Filter": lee_result,
        "Frost Filter": frost_result,
        "CLAHE": clahe_result,
        "ADMM-TV": tv_result,
    }

    fig = plot_preprocessing_comparison(gray, processed, figsize=(18, 4))
    fig.savefig(output_dir / "preprocessing_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n   Saved: {output_dir / 'preprocessing_comparison.png'}")

    fig = plot_speckle_analysis(gray)
    fig.savefig(output_dir / "speckle_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {output_dir / 'speckle_analysis.png'}")

    return processed


def demo_segmentation(output_dir: Path) -> None:
    """Demonstrate segmentation models."""
    _ensure_output_dir(output_dir)
    print("\n" + "=" * 60)
    print("SEGMENTATION MODEL DEMONSTRATION")
    print("=" * 60)


    model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512])
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nU-Net Architecture:")
    print("-" * 40)
    print("   Input channels: 3 (RGB)")
    print("   Output channels: 1 (binary mask)")
    print("   Feature channels: [64, 128, 256, 512]")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(x)

    print(f"\n   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")

    print("\nLoss Functions:")
    print("-" * 40)

    pred = torch.randn(1, 1, 256, 256)
    target = torch.randint(0, 2, (1, 1, 256, 256)).float()

    dice = dice_loss(pred, target)
    combined = combined_loss(pred, target, bce_weight=0.5)

    print(f"   Dice Loss: {dice.item():.4f}")
    print(f"   Combined (BCE + Dice) Loss: {combined.item():.4f}")

    attn_model = AttentionUNet(in_channels=3, out_channels=1)
    attn_params = sum(p.numel() for p in attn_model.parameters())

    print("\nAttention U-Net:")
    print("-" * 40)
    print(f"   Total parameters: {attn_params:,}")
    print("   Includes attention gates for focused feature learning")


def demo_classification(output_dir: Path) -> None:
    """Demonstrate classification models."""
    _ensure_output_dir(output_dir)
    print("\n" + "=" * 60)
    print("CLASSIFICATION MODEL DEMONSTRATION")
    print("=" * 60)


    custom_model = UltrasoundClassifier(num_classes=2, dropout=0.5)
    custom_params = sum(p.numel() for p in custom_model.parameters())

    print("\nCustom CNN Classifier:")
    print("-" * 40)
    print("   Classes: 2 (benign, malignant)")
    print(f"   Parameters: {custom_params:,}")

    pretrained_backbone = True
    try:
        resnet_model = ResNetClassifier(num_classes=2, pretrained=True, model_name="resnet18")
    except Exception as exc:
        pretrained_backbone = False
        print(f"   Pretrained weights unavailable ({exc}); falling back to random initialization")
        resnet_model = ResNetClassifier(num_classes=2, pretrained=False, model_name="resnet18")
    resnet_params = sum(p.numel() for p in resnet_model.parameters())
    resnet_trainable = sum(p.numel() for p in resnet_model.parameters() if p.requires_grad)

    print("\nResNet-18 Transfer Learning:")
    print("-" * 40)
    if pretrained_backbone:
        print("   Pretrained on ImageNet")
    else:
        print("   Using randomly initialized backbone")
    print(f"   Total parameters: {resnet_params:,}")
    print(f"   Trainable parameters: {resnet_trainable:,}")
    print("   Frozen backbone for initial training")

    x = torch.randn(4, 3, 256, 256)
    with torch.no_grad():
        custom_out = custom_model(x)
        resnet_out = resnet_model(x)

    print("\n   Batch size: 4, Input: 256x256 RGB")
    print(f"   Custom CNN output: {custom_out.shape}")
    print(f"   ResNet output: {resnet_out.shape}")


def demo_admm_optimization(image: np.ndarray, output_dir: Path) -> None:
    """Detailed ADMM optimization demo for TV denoising."""
    _ensure_output_dir(output_dir)
    print("\n" + "=" * 60)
    print("ADMM OPTIMIZATION DEMONSTRATION")
    print("=" * 60)

    gray = np.mean(image, axis=2).astype(np.uint8) if image.ndim == 3 else image
    lambdas = [0.01, 0.05, 0.1, 0.2]

    fig, axes = plt.subplots(2, len(lambdas) + 1, figsize=(16, 8))

    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    print("\nADMM Convergence Analysis:")
    print("-" * 40)

    for i, lam in enumerate(lambdas):
        result, conv = admm_tv_denoising(gray, lambda_tv=lam, rho=1.0, n_iter=50, verbose=False)

        axes[0, i + 1].imshow(result, cmap="gray")
        axes[0, i + 1].set_title(f"λ = {lam}")
        axes[0, i + 1].axis("off")

        axes[1, i + 1].semilogy(conv["primal_residuals"], label="Primal", color="blue")
        axes[1, i + 1].semilogy(conv["dual_residuals"], label="Dual", color="red")
        axes[1, i + 1].set_xlabel("Iteration")
        axes[1, i + 1].set_ylabel("Residual")
        axes[1, i + 1].legend()
        axes[1, i + 1].grid(True, alpha=0.3)

        final_primal = conv["primal_residuals"][-1]
        final_dual = conv["dual_residuals"][-1]
        print(f"   λ={lam}: Final primal={final_primal:.6f}, dual={final_dual:.6f}")

    plt.suptitle("ADMM Total Variation Denoising - Parameter Study", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "admm_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n   Saved: {output_dir / 'admm_analysis.png'}")


def demo_full_pipeline(data_dir: Path, output_dir: Path) -> None:
    """Demonstrate an end-to-end dataset loading and visualization flow."""
    _ensure_output_dir(output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60)
    print("FULL PIPELINE DEMONSTRATION")
    print("=" * 60)

    if not (data_dir / "benign").exists():
        print("\nCreating synthetic ultrasound data for demonstration...")
        create_sample_data(str(data_dir), num_samples=5)


    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    mask_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )

    dataset = BUSIDataset(
        root_dir=str(data_dir),
        split="train",
        transform=transform,
        mask_transform=mask_transform,
    )

    print("\nDataset loaded:")
    print(f"   Samples: {len(dataset)}")

    if len(dataset) > 0:
        image, mask, label = dataset[0]

        print(f"   Image shape: {image.shape}")
        print(f"   Mask shape: {mask.shape}")
        print(f"   Label: {'Malignant' if label == 1 else 'Benign'}")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        img_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().numpy()

        axes[0].imshow(img_np)
        axes[0].set_title("Ultrasound Image")
        axes[0].axis("off")

        axes[1].imshow(mask_np, cmap="gray")
        axes[1].set_title("Segmentation Mask")
        axes[1].axis("off")

        overlay = img_np.copy()
        overlay[mask_np > 0.5] = [1, 0, 0]
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay ({'Malignant' if label == 1 else 'Benign'})")
        axes[2].axis("off")

        plt.tight_layout()
        fig.savefig(output_dir / "sample_data.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n   Saved: {output_dir / 'sample_data.png'}")


AVAILABLE_DEMOS = ["preprocessing", "admm", "segmentation", "classification", "pipeline"]


def main(demos: Sequence[str], output_dir: Path, data_dir: Path) -> None:
    """Run selected demonstrations."""
    print("=" * 60)
    print("ULTRASOUND IMAGING TOOLKIT")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    test_image: np.ndarray | None = None

    def _get_test_image() -> np.ndarray:
        nonlocal test_image
        if test_image is None:

            print("\nGenerating synthetic ultrasound image for demonstration...")
            test_image = _generate_synthetic_ultrasound("benign", size=(256, 256))
            Image.fromarray(test_image).save(output_dir / "synthetic_ultrasound.png")
            print(f"Saved: {output_dir / 'synthetic_ultrasound.png'}")
        return test_image

    if "preprocessing" in demos:
        demo_preprocessing(_get_test_image(), output_dir)
    if "admm" in demos:
        demo_admm_optimization(_get_test_image(), output_dir)
    if "segmentation" in demos:
        demo_segmentation(output_dir)
    if "classification" in demos:
        demo_classification(output_dir)
    if "pipeline" in demos:
        demo_full_pipeline(data_dir, output_dir)

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ultrasound Imaging Toolkit - Demo Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # run all demos
  python main.py --demo preprocessing admm    # run specific demos
  python main.py --output-dir results         # custom output directory
""",
    )
    parser.add_argument(
        "--demo",
        nargs="+",
        choices=AVAILABLE_DEMOS,
        default=AVAILABLE_DEMOS,
        help="Demo(s) to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save outputs (default: outputs)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/busi"),
        help="Directory for dataset (default: data/busi)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    main(demos=args.demo, output_dir=args.output_dir, data_dir=args.data_dir)
