"""
Evaluation Metrics for Ultrasound Image Analysis.

Includes metrics for:
- Segmentation: Dice, IoU, Hausdorff distance
- Classification: Accuracy, AUC-ROC, confusion matrix
"""

import numpy as np
from typing import Union, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sklearn_confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    classification_report,
)


def compute_dice(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """
    Compute Dice Similarity Coefficient (DSC).
    
    DSC = 2 * |A ∩ B| / (|A| + |B|)
    
    Measures overlap between prediction and ground truth.
    Range: [0, 1], higher is better.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        smooth: Smoothing factor for numerical stability
        
    Returns:
        Dice coefficient
    """
    pred = pred.flatten().astype(bool)
    target = target.flatten().astype(bool)
    
    intersection = np.sum(pred & target)
    dice = (2 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)
    
    return float(dice)


def compute_iou(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard Index).
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        smooth: Smoothing factor
        
    Returns:
        IoU score
    """
    pred = pred.flatten().astype(bool)
    target = target.flatten().astype(bool)
    
    intersection = np.sum(pred & target)
    union = np.sum(pred | target)
    
    iou = (intersection + smooth) / (union + smooth)
    
    return float(iou)


def compute_hausdorff_distance(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Compute 95th percentile Hausdorff Distance.
    
    Measures the boundary distance between prediction and ground truth.
    Lower is better.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        
    Returns:
        Hausdorff distance (95th percentile)
    """
    from scipy.ndimage import distance_transform_edt
    
    # Find boundary points
    pred_boundary = pred ^ np.roll(pred, 1, axis=0) | pred ^ np.roll(pred, 1, axis=1)
    target_boundary = target ^ np.roll(target, 1, axis=0) | target ^ np.roll(target, 1, axis=1)
    
    # Distance transforms
    dist_pred = distance_transform_edt(~target)
    dist_target = distance_transform_edt(~pred)
    
    # Get distances at boundary points
    if np.any(pred_boundary):
        d1 = dist_pred[pred_boundary]
    else:
        d1 = np.array([0])
        
    if np.any(target_boundary):
        d2 = dist_target[target_boundary]
    else:
        d2 = np.array([0])
    
    # 95th percentile Hausdorff distance
    hd95 = max(np.percentile(d1, 95), np.percentile(d2, 95))
    
    return float(hd95)


def compute_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        pred: Predicted labels
        target: Ground truth labels
        
    Returns:
        Accuracy score
    """
    return float(accuracy_score(target.flatten(), pred.flatten()))


def compute_confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        pred: Predicted labels
        target: Ground truth labels
        normalize: Normalization mode ('true', 'pred', 'all', or None)
        
    Returns:
        Confusion matrix
    """
    return sklearn_confusion_matrix(target.flatten(), pred.flatten(), normalize=normalize)


def compute_auc_roc(
    pred_proba: np.ndarray,
    target: np.ndarray,
    multi_class: str = 'ovr',
) -> float:
    """
    Compute Area Under the ROC Curve.
    
    Args:
        pred_proba: Predicted probabilities
        target: Ground truth labels
        multi_class: Strategy for multi-class ('ovr' or 'ovo')
        
    Returns:
        AUC-ROC score
    """
    # Handle binary classification
    if pred_proba.ndim == 1 or pred_proba.shape[1] == 2:
        if pred_proba.ndim == 2:
            pred_proba = pred_proba[:, 1]
        return float(roc_auc_score(target, pred_proba))
    
    # Multi-class
    return float(roc_auc_score(target, pred_proba, multi_class=multi_class))


def compute_classification_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    class_names: Optional[list] = None,
) -> dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        pred: Predicted labels
        target: Ground truth labels
        class_names: Names of classes
        
    Returns:
        Dictionary with precision, recall, F1, etc.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        target, pred, average=None
    )
    
    # Macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        target, pred, average='macro'
    )
    
    metrics = {
        'accuracy': float(accuracy_score(target, pred)),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'support_per_class': support.tolist(),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
    }
    
    if class_names:
        metrics['class_names'] = class_names
    
    return metrics


def compute_segmentation_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> dict:
    """
    Compute comprehensive segmentation metrics.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        
    Returns:
        Dictionary with Dice, IoU, Hausdorff, etc.
    """
    # Binarize if needed
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    
    return {
        'dice': compute_dice(pred, target),
        'iou': compute_iou(pred, target),
        'hausdorff_95': compute_hausdorff_distance(pred, target),
        'pixel_accuracy': compute_accuracy(pred, target),
        'sensitivity': compute_sensitivity(pred, target),
        'specificity': compute_specificity(pred, target),
    }


def compute_sensitivity(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Compute sensitivity (True Positive Rate / Recall).
    
    Sensitivity = TP / (TP + FN)
    """
    pred = pred.flatten().astype(bool)
    target = target.flatten().astype(bool)
    
    tp = np.sum(pred & target)
    fn = np.sum(~pred & target)
    
    return float(tp / (tp + fn + 1e-10))


def compute_specificity(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Compute specificity (True Negative Rate).
    
    Specificity = TN / (TN + FP)
    """
    pred = pred.flatten().astype(bool)
    target = target.flatten().astype(bool)
    
    tn = np.sum(~pred & ~target)
    fp = np.sum(pred & ~target)
    
    return float(tn / (tn + fp + 1e-10))
