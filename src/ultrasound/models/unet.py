"""
U-Net Architecture for Ultrasound Image Segmentation.

U-Net is a convolutional neural network architecture designed for
biomedical image segmentation. It consists of:
- Encoder (contracting path): Captures context through downsampling
- Decoder (expansive path): Enables precise localization through upsampling
- Skip connections: Combine low-level and high-level features

Reference:
    Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional 
    Networks for Biomedical Image Segmentation. MICCAI 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ConvBlock(nn.Module):
    """Double convolution block: (Conv -> BN -> ReLU) x 2"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block: MaxPool -> ConvBlock"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upsampling block: Upsample -> Concat -> ConvBlock"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels, dropout)
            
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle size mismatch due to padding
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for ultrasound image segmentation.
    
    Architecture:
        - 4 encoder blocks with increasing feature channels
        - Bottleneck
        - 4 decoder blocks with skip connections
        - Final 1x1 convolution for segmentation map
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        out_channels: Number of output classes
        features: List of feature channels for each level
        bilinear: Use bilinear upsampling instead of transposed convolution
        dropout: Dropout rate for regularization
        
    Example:
        >>> model = UNet(in_channels=3, out_channels=1)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> output = model(x)  # Shape: [1, 1, 256, 256]
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
        bilinear: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.bilinear = bilinear
        
        # Input convolution
        self.inc = ConvBlock(in_channels, features[0])
        
        # Encoder (downsampling)
        self.down1 = DownBlock(features[0], features[1], dropout)
        self.down2 = DownBlock(features[1], features[2], dropout)
        self.down3 = DownBlock(features[2], features[3], dropout)
        
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(features[3], features[3] * 2 // factor, dropout)
        
        # Decoder (upsampling)
        self.up1 = UpBlock(features[3] * 2, features[3] // factor, bilinear, dropout)
        self.up2 = UpBlock(features[3], features[2] // factor, bilinear, dropout)
        self.up3 = UpBlock(features[2], features[1] // factor, bilinear, dropout)
        self.up4 = UpBlock(features[1], features[0], bilinear, dropout)
        
        # Output convolution
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)      # 64 channels
        x2 = self.down1(x1)   # 128 channels
        x3 = self.down2(x2)   # 256 channels
        x4 = self.down3(x3)   # 512 channels
        x5 = self.down4(x4)   # 1024 or 512 channels (bottleneck)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict binary segmentation mask."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()


class UNetSmall(nn.Module):
    """
    Smaller U-Net variant for faster training and inference.
    
    Useful for prototyping and smaller datasets.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256],
        bilinear: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            bilinear=bilinear,
            dropout=dropout,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AttentionGate(nn.Module):
    """
    Attention gate for focusing on salient features.
    
    Reference:
        Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look 
        for the Pancreas.
    """
    
    def __init__(
        self,
        gate_channels: int,
        in_channels: int,
        inter_channels: Optional[int] = None,
    ):
        super().__init__()
        
        if inter_channels is None:
            inter_channels = in_channels // 2
            
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g to match x size
        g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class AttentionUNet(nn.Module):
    """
    Attention U-Net for ultrasound image segmentation.
    
    Adds attention gates to standard U-Net for better focus on
    relevant features while suppressing irrelevant regions.
    
    This is particularly useful for ultrasound images where
    lesions may be subtle and surrounded by noisy speckle.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
        bilinear: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.features = features
        
        # Encoder
        self.inc = ConvBlock(in_channels, features[0])
        self.down1 = DownBlock(features[0], features[1], dropout)
        self.down2 = DownBlock(features[1], features[2], dropout)
        self.down3 = DownBlock(features[2], features[3], dropout)
        
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(features[3], features[3] * 2 // factor, dropout)
        
        # Attention gates
        self.att1 = AttentionGate(features[3] * 2 // factor, features[3])
        self.att2 = AttentionGate(features[3] // factor, features[2])
        self.att3 = AttentionGate(features[2] // factor, features[1])
        self.att4 = AttentionGate(features[1] // factor, features[0])
        
        # Decoder
        self.up1 = UpBlock(features[3] * 2, features[3] // factor, bilinear, dropout)
        self.up2 = UpBlock(features[3], features[2] // factor, bilinear, dropout)
        self.up3 = UpBlock(features[2], features[1] // factor, bilinear, dropout)
        self.up4 = UpBlock(features[1], features[0], bilinear, dropout)
        
        # Output
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with attention
        x4 = self.att1(x5, x4)
        d4 = self.up1(x5, x4)
        
        x3 = self.att2(d4, x3)
        d3 = self.up2(d4, x3)
        
        x2 = self.att3(d3, x2)
        d2 = self.up3(d3, x2)
        
        x1 = self.att4(d2, x1)
        d1 = self.up4(d2, x1)
        
        return self.outc(d1)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Dice loss for segmentation.
    
    Dice coefficient measures overlap between prediction and ground truth.
    Dice loss = 1 - Dice coefficient
    
    Args:
        pred: Predicted probabilities
        target: Ground truth binary mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice loss value
    """
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
) -> torch.Tensor:
    """
    Combined BCE + Dice loss for segmentation.
    
    Args:
        pred: Predicted logits
        target: Ground truth binary mask
        bce_weight: Weight for BCE loss (1 - bce_weight for Dice)
        
    Returns:
        Combined loss value
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    
    return bce_weight * bce + (1 - bce_weight) * dice
