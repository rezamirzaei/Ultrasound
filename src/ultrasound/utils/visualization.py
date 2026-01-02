"""
Visualization utilities for ultrasound image analysis.

Provides functions for:
- Preprocessing comparison visualizations
- Segmentation result overlays
- Training progress plots
- ROC curves and confusion matrices
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Optional, Tuple, Dict, Union
from pathlib import Path


# Custom colormap for ultrasound visualization
ULTRASOUND_CMAP = LinearSegmentedColormap.from_list(
    'ultrasound', 
    [(0, 0, 0), (0.2, 0.2, 0.2), (0.5, 0.5, 0.5), (0.8, 0.8, 0.8), (1, 1, 1)]
)


def visualize_results(
    images: List[np.ndarray],
    titles: List[str],
    figsize: Tuple[int, int] = (15, 5),
    cmap: str = 'gray',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize multiple images side by side.
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        figsize: Figure size
        cmap: Colormap for display
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 3 and img.shape[2] == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_preprocessing_comparison(
    original: np.ndarray,
    processed: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (16, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare original and preprocessed ultrasound images.
    
    Args:
        original: Original ultrasound image
        processed: Dictionary of {method_name: processed_image}
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    n = len(processed) + 1
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    # Original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original', fontweight='bold')
    axes[0].axis('off')
    
    # Processed images
    for ax, (name, img) in zip(axes[1:], processed.items()):
        ax.imshow(img, cmap='gray')
        ax.set_title(name, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Ultrasound Image Preprocessing Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_segmentation_overlay(
    image: np.ndarray,
    mask_true: np.ndarray,
    mask_pred: Optional[np.ndarray] = None,
    alpha: float = 0.4,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize segmentation results with overlay.
    
    Args:
        image: Original image
        mask_true: Ground truth mask
        mask_pred: Predicted mask (optional)
        alpha: Overlay transparency
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if mask_pred is not None:
        fig, axes = plt.subplots(1, 4, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Convert grayscale to RGB for overlay
    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image.copy()
    
    # Normalize to [0, 1]
    if image_rgb.max() > 1:
        image_rgb = image_rgb / 255.0
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth overlay
    overlay_true = create_mask_overlay(image_rgb, mask_true, color=[0, 1, 0], alpha=alpha)
    axes[1].imshow(overlay_true)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    if mask_pred is not None:
        # Prediction overlay
        overlay_pred = create_mask_overlay(image_rgb, mask_pred, color=[1, 0, 0], alpha=alpha)
        axes[2].imshow(overlay_pred)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Comparison overlay (green=TP, red=FP, blue=FN)
        comparison = create_comparison_overlay(image_rgb, mask_true, mask_pred, alpha)
        axes[3].imshow(comparison)
        axes[3].set_title('Comparison\n(Green=TP, Red=FP, Blue=FN)')
        axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: List[float] = [1, 0, 0],
    alpha: float = 0.4,
) -> np.ndarray:
    """Create overlay of mask on image."""
    overlay = image.copy()
    mask_bool = mask > 0.5
    
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
            overlay[:, :, c]
        )
    
    return np.clip(overlay, 0, 1)


def create_comparison_overlay(
    image: np.ndarray,
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Create comparison overlay showing TP, FP, FN."""
    overlay = image.copy()
    
    true_bool = mask_true > 0.5
    pred_bool = mask_pred > 0.5
    
    tp = true_bool & pred_bool  # True Positive - Green
    fp = pred_bool & ~true_bool  # False Positive - Red
    fn = true_bool & ~pred_bool  # False Negative - Blue
    
    colors = {
        'tp': [0, 1, 0],  # Green
        'fp': [1, 0, 0],  # Red
        'fn': [0, 0, 1],  # Blue
    }
    
    for mask_type, mask in [('tp', tp), ('fp', fp), ('fn', fn)]:
        color = colors[mask_type]
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c]
            )
    
    return np.clip(overlay, 0, 1)


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = ['loss', 'accuracy'],
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with metric names and values per epoch
        metrics: List of metrics to plot
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric in history:
            ax.plot(history[metric], label=f'Train {metric}', color='blue')
        if f'val_{metric}' in history:
            ax.plot(history[f'val_{metric}'], label=f'Val {metric}', color='orange')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curve for classification.
    
    Args:
        y_true: True labels
        y_score: Predicted probabilities
        class_names: Names of classes
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Binary classification
    if y_score.ndim == 1 or y_score.shape[1] == 2:
        if y_score.ndim == 2:
            y_score = y_score[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:
        # Multi-class
        from sklearn.preprocessing import label_binarize
        n_classes = y_score.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        
        for i, color in enumerate(colors):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            
            name = class_names[i] if class_names else f'Class {i}'
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        normalize: Normalize to percentages
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_speckle_analysis(
    image: np.ndarray,
    window_size: int = 32,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Analyze and visualize speckle characteristics in ultrasound image.
    
    Args:
        image: Input ultrasound image
        window_size: Size of local analysis window
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if image.ndim == 3:
        image = image[:, :, 0]
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Histogram
    axes[1].hist(image.flatten(), bins=50, color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Intensity')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Intensity Histogram')
    
    # Local coefficient of variation map
    from scipy.ndimage import uniform_filter
    
    local_mean = uniform_filter(image.astype(float), size=window_size)
    local_sq_mean = uniform_filter(image.astype(float) ** 2, size=window_size)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
    cv_map = local_std / (local_mean + 1e-10)
    
    im = axes[2].imshow(cv_map, cmap='hot')
    axes[2].set_title('Coefficient of Variation')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    # Speckle statistics
    mean_val = np.mean(image)
    std_val = np.std(image)
    cv_global = std_val / mean_val
    
    stats_text = f"""
    Speckle Statistics:
    
    Mean intensity: {mean_val:.2f}
    Std deviation: {std_val:.2f}
    Global CV: {cv_global:.3f}
    
    For fully developed speckle,
    CV ≈ 1.0 (Rayleigh)
    
    Image CV: {cv_global:.3f}
    """
    
    axes[3].text(0.1, 0.5, stats_text, transform=axes[3].transAxes,
                fontsize=11, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[3].axis('off')
    axes[3].set_title('Speckle Analysis')
    
    plt.suptitle('Ultrasound Speckle Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
