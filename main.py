"""
Ultrasound Imaging Toolkit - Demo Script
=========================================

This script demonstrates the capabilities of the ultrasound imaging toolkit:
1. Loading and preprocessing ultrasound images
2. Speckle reduction techniques
3. Contrast enhancement methods
4. ADMM-based Total Variation denoising
5. Deep learning segmentation with U-Net
6. Classification of breast lesions

Author: Reza Mirzaeifard, PhD
Email: reza.mirzaeifard@gmail.com

For InPhase Solutions AS - Senior Consultant Position Application
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ultrasound.data import BUSIDataset, create_sample_data, download_busi_dataset
from ultrasound.preprocessing.speckle import SpeckleReducer, lee_filter, frost_filter
from ultrasound.preprocessing.enhancement import ContrastEnhancer, apply_clahe
from ultrasound.preprocessing.denoising import admm_tv_denoising, total_variation_denoising
from ultrasound.preprocessing.normalization import normalize_image, depth_compensation
from ultrasound.models.unet import UNet, UNetSmall, dice_loss, combined_loss
from ultrasound.models.classifier import UltrasoundClassifier, ResNetClassifier
from ultrasound.utils.metrics import compute_dice, compute_iou, compute_segmentation_metrics
from ultrasound.utils.visualization import (
    visualize_results,
    plot_preprocessing_comparison,
    plot_segmentation_overlay,
    plot_speckle_analysis,
)


def demo_preprocessing(image: np.ndarray, output_dir: Path):
    """
    Demonstrate various preprocessing techniques for ultrasound images.
    
    This showcases:
    - Speckle reduction (Lee, Frost, Median filters)
    - Contrast enhancement (CLAHE, histogram equalization)
    - ADMM-based Total Variation denoising
    """
    print("\n" + "="*60)
    print("PREPROCESSING DEMONSTRATION")
    print("="*60)
    
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray = image
    
    print("\n1. Speckle Reduction Techniques:")
    print("-" * 40)
    
    # Lee filter
    reducer_lee = SpeckleReducer(method='lee', window_size=7)
    lee_result = reducer_lee.reduce(gray)
    mean_val, cv = reducer_lee.estimate_speckle_level(gray)
    print(f"   Original - Mean: {mean_val:.2f}, CV: {cv:.3f}")
    mean_val, cv = reducer_lee.estimate_speckle_level(lee_result)
    print(f"   Lee Filter - Mean: {mean_val:.2f}, CV: {cv:.3f}")
    
    # Frost filter  
    reducer_frost = SpeckleReducer(method='frost', window_size=5, damping_factor=1.5)
    frost_result = reducer_frost.reduce(gray)
    
    # Median filter
    reducer_median = SpeckleReducer(method='median', window_size=5)
    median_result = reducer_median.reduce(gray)
    
    print("\n2. Contrast Enhancement:")
    print("-" * 40)
    
    # CLAHE
    enhancer = ContrastEnhancer(method='clahe', clip_limit=2.5)
    clahe_result = enhancer.enhance(gray)
    
    # Analyze contrast
    original_stats = enhancer.analyze_contrast(gray)
    enhanced_stats = enhancer.analyze_contrast(clahe_result)
    print(f"   Original contrast ratio: {original_stats['contrast_ratio']:.3f}")
    print(f"   CLAHE contrast ratio: {enhanced_stats['contrast_ratio']:.3f}")
    
    print("\n3. ADMM-based Total Variation Denoising:")
    print("-" * 40)
    
    # ADMM TV denoising (showcases optimization expertise)
    tv_result, convergence = admm_tv_denoising(
        gray, 
        lambda_tv=0.05, 
        rho=1.0, 
        n_iter=30,
        verbose=False
    )
    print(f"   Converged in {len(convergence['primal_residuals'])} iterations")
    print(f"   Final primal residual: {convergence['primal_residuals'][-1]:.6f}")
    
    # Plot comparison
    processed = {
        'Lee Filter': lee_result,
        'Frost Filter': frost_result,
        'CLAHE': clahe_result,
        'ADMM-TV': tv_result,
    }
    
    fig = plot_preprocessing_comparison(gray, processed, figsize=(18, 4))
    fig.savefig(output_dir / 'preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n   Saved: {output_dir / 'preprocessing_comparison.png'}")
    
    # Speckle analysis
    fig = plot_speckle_analysis(gray)
    fig.savefig(output_dir / 'speckle_analysis.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'speckle_analysis.png'}")
    
    return processed


def demo_segmentation(output_dir: Path):
    """
    Demonstrate U-Net segmentation for ultrasound images.
    
    Shows the U-Net architecture designed for medical image segmentation.
    """
    print("\n" + "="*60)
    print("SEGMENTATION MODEL DEMONSTRATION")
    print("="*60)
    
    import torch
    
    # Create model
    model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512])
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nU-Net Architecture:")
    print("-" * 40)
    print(f"   Input channels: 3 (RGB)")
    print(f"   Output channels: 1 (binary mask)")
    print(f"   Feature channels: [64, 128, 256, 512]")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Forward pass with dummy input
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
    
    print(f"\n   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Demonstrate loss functions
    print(f"\nLoss Functions:")
    print("-" * 40)
    
    pred = torch.randn(1, 1, 256, 256)
    target = torch.randint(0, 2, (1, 1, 256, 256)).float()
    
    dice = dice_loss(pred, target)
    combined = combined_loss(pred, target, bce_weight=0.5)
    
    print(f"   Dice Loss: {dice.item():.4f}")
    print(f"   Combined (BCE + Dice) Loss: {combined.item():.4f}")
    
    # Attention U-Net
    from ultrasound.models.unet import AttentionUNet
    
    attn_model = AttentionUNet(in_channels=3, out_channels=1)
    attn_params = sum(p.numel() for p in attn_model.parameters())
    
    print(f"\nAttention U-Net:")
    print("-" * 40)
    print(f"   Total parameters: {attn_params:,}")
    print(f"   Includes attention gates for focused feature learning")


def demo_classification(output_dir: Path):
    """
    Demonstrate classification models for breast ultrasound.
    """
    print("\n" + "="*60)
    print("CLASSIFICATION MODEL DEMONSTRATION")
    print("="*60)
    
    import torch
    
    # Custom classifier
    custom_model = UltrasoundClassifier(num_classes=2, dropout=0.5)
    custom_params = sum(p.numel() for p in custom_model.parameters())
    
    print(f"\nCustom CNN Classifier:")
    print("-" * 40)
    print(f"   Classes: 2 (benign, malignant)")
    print(f"   Parameters: {custom_params:,}")
    
    # ResNet transfer learning
    resnet_model = ResNetClassifier(num_classes=2, pretrained=True, model_name='resnet18')
    resnet_params = sum(p.numel() for p in resnet_model.parameters())
    resnet_trainable = sum(p.numel() for p in resnet_model.parameters() if p.requires_grad)
    
    print(f"\nResNet-18 Transfer Learning:")
    print("-" * 40)
    print(f"   Pretrained on ImageNet")
    print(f"   Total parameters: {resnet_params:,}")
    print(f"   Trainable parameters: {resnet_trainable:,}")
    print(f"   Frozen backbone for initial training")
    
    # Test forward pass
    x = torch.randn(4, 3, 256, 256)
    with torch.no_grad():
        custom_out = custom_model(x)
        resnet_out = resnet_model(x)
    
    print(f"\n   Batch size: 4, Input: 256x256 RGB")
    print(f"   Custom CNN output: {custom_out.shape}")
    print(f"   ResNet output: {resnet_out.shape}")


def demo_admm_optimization(image: np.ndarray, output_dir: Path):
    """
    Detailed demonstration of ADMM optimization for image denoising.
    
    This showcases my PhD expertise in:
    - Alternating Direction Method of Multipliers (ADMM)
    - Proximal operators (soft thresholding)
    - Total Variation regularization
    - Convergence analysis
    """
    print("\n" + "="*60)
    print("ADMM OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    print("""
    ADMM for Total Variation Denoising
    -----------------------------------
    
    Problem formulation:
        min_u  (1/2)||u - f||² + λ||Du||₁
    
    where:
        u: denoised image
        f: noisy input image  
        D: gradient operator
        λ: regularization weight
    
    ADMM reformulation introduces auxiliary variable z = Du:
        min_u,z  (1/2)||u - f||² + λ||z||₁
        s.t.     Du = z
    
    The ADMM iterations are:
        u^{k+1} = (I + ρD^TD)^{-1}(f + D^T(ρz^k - y^k))
        z^{k+1} = S_{λ/ρ}(Du^{k+1} + y^k/ρ)   [soft thresholding]
        y^{k+1} = y^k + ρ(Du^{k+1} - z^{k+1})
    """)
    
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray = image
    
    # Run ADMM with different parameters
    lambdas = [0.01, 0.05, 0.1, 0.2]
    
    fig, axes = plt.subplots(2, len(lambdas) + 1, figsize=(16, 8))
    
    # Original
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    print("\nADMM Convergence Analysis:")
    print("-" * 40)
    
    for i, lam in enumerate(lambdas):
        result, conv = admm_tv_denoising(
            gray,
            lambda_tv=lam,
            rho=1.0,
            n_iter=50,
            verbose=False
        )
        
        # Show result
        axes[0, i+1].imshow(result, cmap='gray')
        axes[0, i+1].set_title(f'λ = {lam}')
        axes[0, i+1].axis('off')
        
        # Show convergence
        axes[1, i+1].semilogy(conv['primal_residuals'], label='Primal', color='blue')
        axes[1, i+1].semilogy(conv['dual_residuals'], label='Dual', color='red')
        axes[1, i+1].set_xlabel('Iteration')
        axes[1, i+1].set_ylabel('Residual')
        axes[1, i+1].legend()
        axes[1, i+1].grid(True, alpha=0.3)
        
        final_primal = conv['primal_residuals'][-1]
        final_dual = conv['dual_residuals'][-1]
        print(f"   λ={lam}: Final primal={final_primal:.6f}, dual={final_dual:.6f}")
    
    plt.suptitle('ADMM Total Variation Denoising - Parameter Study', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'admm_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n   Saved: {output_dir / 'admm_analysis.png'}")


def demo_full_pipeline(data_dir: Path, output_dir: Path):
    """
    Demonstrate full pipeline from data loading to evaluation.
    """
    print("\n" + "="*60)
    print("FULL PIPELINE DEMONSTRATION")
    print("="*60)
    
    # Create synthetic data if real data not available
    if not (data_dir / 'benign').exists():
        print("\nCreating synthetic ultrasound data for demonstration...")
        create_sample_data(str(data_dir), num_samples=5)
    
    # Load dataset
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = BUSIDataset(
        root_dir=str(data_dir),
        split='train',
        transform=transform,
        mask_transform=mask_transform,
    )
    
    print(f"\nDataset loaded:")
    print(f"   Samples: {len(dataset)}")
    
    if len(dataset) > 0:
        # Get sample
        image, mask, label = dataset[0]
        
        print(f"   Image shape: {image.shape}")
        print(f"   Mask shape: {mask.shape}")
        print(f"   Label: {'Malignant' if label == 1 else 'Benign'}")
        
        # Visualize sample
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        img_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().numpy()
        
        axes[0].imshow(img_np)
        axes[0].set_title('Ultrasound Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # Overlay
        overlay = img_np.copy()
        overlay[mask_np > 0.5] = [1, 0, 0]  # Red overlay
        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay ({"Malignant" if label == 1 else "Benign"})')
        axes[2].axis('off')
        
        plt.tight_layout()
        fig.savefig(output_dir / 'sample_data.png', dpi=150, bbox_inches='tight')
        print(f"\n   Saved: {output_dir / 'sample_data.png'}")


def main():
    """Main demo function."""
    print("="*60)
    print("ULTRASOUND IMAGING TOOLKIT")
    print("Demo for InPhase Solutions AS Application")
    print("="*60)
    print("\nAuthor: Reza Mirzaeifard, PhD")
    print("Email: reza.mirzaeifard@gmail.com")
    print("\nThis toolkit demonstrates expertise in:")
    print("  • Ultrasound image preprocessing")
    print("  • Speckle noise reduction")
    print("  • Signal/image processing algorithms")
    print("  • Machine learning for medical imaging")
    print("  • Optimization-based methods (ADMM)")
    
    # Setup directories
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    data_dir = Path('data/busi')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic test image
    print("\n\nGenerating synthetic ultrasound image for demonstration...")
    from ultrasound.data import _generate_synthetic_ultrasound
    test_image = _generate_synthetic_ultrasound('benign', size=(256, 256))
    
    # Save synthetic image
    from PIL import Image
    Image.fromarray(test_image).save(output_dir / 'synthetic_ultrasound.png')
    print(f"Saved: {output_dir / 'synthetic_ultrasound.png'}")
    
    # Run demonstrations
    demo_preprocessing(test_image, output_dir)
    demo_admm_optimization(test_image, output_dir)
    demo_segmentation(output_dir)
    demo_classification(output_dir)
    demo_full_pipeline(data_dir, output_dir)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print("\nTo use with real data:")
    print("  1. Download BUSI dataset from Kaggle:")
    print("     https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
    print("  2. Extract to data/busi/")
    print("  3. Run the pipeline again")
    print("\nFor questions: reza.mirzaeifard@gmail.com")


if __name__ == '__main__':
    main()

