"""
Classification Models for Ultrasound Images.

This module provides classifiers for ultrasound image classification tasks,
such as distinguishing benign vs malignant breast lesions.

Includes:
- Custom CNN classifier
- Transfer learning with pretrained ResNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple


class UltrasoundClassifier(nn.Module):
    """
    Custom CNN classifier for ultrasound image classification.
    
    A lightweight CNN designed for ultrasound image characteristics:
    - Handles speckle noise patterns
    - Multiple scales of feature extraction
    - Dropout for regularization
    
    Args:
        num_classes: Number of output classes (2 for binary classification)
        in_channels: Number of input channels (3 for RGB)
        dropout: Dropout rate for regularization
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 256 -> 128
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 32 -> 16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5: 16 -> 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


class ResNetClassifier(nn.Module):
    """
    Transfer learning classifier using pretrained ResNet.
    
    Uses ResNet-18 or ResNet-50 pretrained on ImageNet as feature extractor,
    with a custom classification head for ultrasound images.
    
    Transfer learning is effective for medical imaging because:
    - Pretrained features capture general image patterns
    - Fine-tuning adapts to domain-specific features
    - Reduces data requirements for training
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained ImageNet weights
        model_name: ResNet variant ('resnet18', 'resnet34', 'resnet50')
        freeze_backbone: Freeze pretrained layers initially
        dropout: Dropout rate for classifier
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_name: str = 'resnet18',
        freeze_backbone: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_name = model_name
        
        # Load pretrained ResNet
        if model_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            num_features = 512
        elif model_name == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            num_features = 512
        elif model_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace classifier
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the end.
                       If None, unfreeze all layers.
        """
        if num_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last n layers
            layers = list(self.backbone.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
                    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss for handling class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard examples,
    which is useful for imbalanced datasets common in medical imaging.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        pred: Predicted logits
        target: Ground truth labels
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
        
    Returns:
        Focal loss value
    """
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    
    # Apply focal term
    focal_term = (1 - pt) ** gamma
    
    # Apply class balancing
    if alpha is not None:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        focal_loss = alpha_t * focal_term * ce_loss
    else:
        focal_loss = focal_term * ce_loss
    
    return focal_loss.mean()


class EnsembleClassifier(nn.Module):
    """
    Ensemble of multiple classifiers for improved performance.
    
    Combines predictions from multiple models using averaging
    or voting strategies.
    """
    
    def __init__(
        self,
        models: list,
        strategy: str = 'average',
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.strategy = strategy
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs, dim=0)
        
        if self.strategy == 'average':
            return outputs.mean(dim=0)
        elif self.strategy == 'max':
            return outputs.max(dim=0)[0]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
