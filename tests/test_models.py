"""Tests for models (UNet, classifier, and losses)."""

import pytest
import torch

from ultrasound.models.classifier import (
    EnsembleClassifier,
    ResNetClassifier,
    UltrasoundClassifier,
    focal_loss,
)
from ultrasound.models.unet import AttentionUNet, UNet, UNetSmall, combined_loss, dice_loss


class TestUNet:
    def test_output_shape(self):
        model = UNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128])
        x = torch.randn(1, 3, 64, 64)
        assert model(x).shape == (1, 1, 64, 64)

    def test_predict_binary(self):
        model = UNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128])
        x = torch.randn(1, 3, 64, 64)
        mask = model.predict(x)
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})

    def test_features_are_copied_from_input_sequence(self):
        features = [16, 32, 64, 128]
        model = UNet(in_channels=3, out_channels=1, features=features)
        features[0] = 999
        assert model.features == (16, 32, 64, 128)

    def test_invalid_feature_length(self):
        with pytest.raises(ValueError, match="exactly four"):
            UNet(in_channels=3, out_channels=1, features=[16, 32, 64])


class TestUNetSmall:
    def test_output_shape(self):
        model = UNetSmall(in_channels=1, out_channels=1)
        x = torch.randn(2, 1, 64, 64)
        assert model(x).shape == (2, 1, 64, 64)


class TestAttentionUNet:
    def test_output_shape(self):
        model = AttentionUNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128])
        x = torch.randn(1, 3, 64, 64)
        assert model(x).shape == (1, 1, 64, 64)

    def test_invalid_feature_values(self):
        with pytest.raises(ValueError, match="positive integers"):
            AttentionUNet(in_channels=3, out_channels=1, features=[16, 32, 0, 128])


class TestDiceLoss:
    def test_perfect_prediction(self):
        target = torch.ones(1, 1, 8, 8)
        pred = torch.full_like(target, 10.0)  # confident positive logits
        loss = dice_loss(pred, target, from_logits=True)
        assert loss.item() < 0.05

    def test_from_logits_flag(self):
        target = torch.ones(1, 1, 8, 8)
        probs = torch.ones_like(target)
        loss = dice_loss(probs, target, from_logits=False)
        assert loss.item() < 0.05


class TestCombinedLoss:
    def test_returns_scalar(self, torch_seg_pair):
        logits, target = torch_seg_pair
        loss = combined_loss(logits, target)
        assert loss.dim() == 0


class TestUltrasoundClassifier:
    def test_output_shape(self):
        model = UltrasoundClassifier(num_classes=3)
        x = torch.randn(4, 3, 64, 64)
        assert model(x).shape == (4, 3)

    def test_predict(self):
        model = UltrasoundClassifier(num_classes=2)
        x = torch.randn(4, 3, 64, 64)
        assert model.predict(x).shape == (4,)


class TestResNetClassifier:
    def test_output_shape(self):
        model = ResNetClassifier(num_classes=2, pretrained=False)
        x = torch.randn(2, 3, 64, 64)
        assert model(x).shape == (2, 2)

    def test_unfreeze_backbone(self):
        model = ResNetClassifier(num_classes=2, pretrained=False, freeze_backbone=True)
        model.unfreeze_backbone(num_layers=2)
        trainable = sum(1 for p in model.backbone.parameters() if p.requires_grad)
        assert trainable > 0

    def test_invalid_model_name(self):
        with pytest.raises(ValueError):
            ResNetClassifier(model_name="invalid")


class TestFocalLoss:
    def test_returns_scalar(self):
        pred = torch.randn(8, 3)
        target = torch.randint(0, 3, (8,))
        loss = focal_loss(pred, target)
        assert loss.dim() == 0

    def test_no_alpha(self):
        pred = torch.randn(4, 2)
        target = torch.randint(0, 2, (4,))
        loss = focal_loss(pred, target, alpha=None)
        assert loss.item() >= 0


class TestEnsembleClassifier:
    def test_average_strategy(self):
        m1 = UltrasoundClassifier(num_classes=2)
        m2 = UltrasoundClassifier(num_classes=2)
        ens = EnsembleClassifier(models=[m1, m2], strategy="average")
        x = torch.randn(2, 3, 64, 64)
        assert ens(x).shape == (2, 2)

    def test_max_strategy(self):
        m1 = UltrasoundClassifier(num_classes=2)
        m2 = UltrasoundClassifier(num_classes=2)
        ens = EnsembleClassifier(models=[m1, m2], strategy="max")
        x = torch.randn(2, 3, 64, 64)
        assert ens(x).shape == (2, 2)

    def test_invalid_strategy(self):
        m1 = UltrasoundClassifier(num_classes=2)
        ens = EnsembleClassifier(models=[m1], strategy="bad")
        with pytest.raises(ValueError):
            ens(torch.randn(1, 3, 64, 64))
