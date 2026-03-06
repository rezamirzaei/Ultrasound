"""Tests for models (UNet, classifier, and losses)."""

import pytest
import torch
import torch.nn as nn

import ultrasound.models.classifier as classifier_module
from ultrasound.models.classifier import (
    EnsembleClassifier,
    ResNetClassifier,
    UltrasoundClassifier,
    focal_loss,
)
from ultrasound.models.unet import AttentionUNet, UNet, UNetSmall, combined_loss, dice_loss


class _FakeResNetBackbone(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(1, 1)
        self.layer2 = nn.Linear(1, 1)
        self.fc = nn.Linear(features, 2)
        self._features = features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.ones((x.shape[0], self._features), dtype=x.dtype, device=x.device)
        return self.fc(features)


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

    def test_predict_temporarily_uses_eval_mode_and_restores_state(self, monkeypatch):
        model = UltrasoundClassifier(num_classes=2, dropout=0.2)
        model.train()
        observed_training_flags: list[bool] = []

        monkeypatch.setattr(
            model,
            "forward",
            lambda x: observed_training_flags.append(bool(model.training))
            or torch.zeros((x.shape[0], 2)),
        )

        prediction = model.predict(torch.randn(2, 3, 64, 64))

        assert prediction.shape == (2,)
        assert observed_training_flags == [False]
        assert model.training is True

    def test_predict_proba_temporarily_uses_eval_mode_and_restores_state(self, monkeypatch):
        model = UltrasoundClassifier(num_classes=2, dropout=0.2)
        model.train()
        observed_training_flags: list[bool] = []

        monkeypatch.setattr(
            model,
            "forward",
            lambda x: observed_training_flags.append(bool(model.training))
            or torch.tensor([[2.0, 0.0]]).repeat(x.shape[0], 1),
        )

        probabilities = model.predict_proba(torch.randn(2, 3, 64, 64))

        assert probabilities.shape == (2, 2)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
        assert observed_training_flags == [False]
        assert model.training is True

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"num_classes": 0}, "num_classes must be positive"),
            ({"in_channels": 0}, "in_channels must be positive"),
            ({"dropout": 1.0}, "dropout must be in the range"),
        ],
    )
    def test_invalid_init_arguments(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            UltrasoundClassifier(**kwargs)


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

    def test_unfreeze_backbone_rejects_nonpositive_num_layers(self):
        model = ResNetClassifier(num_classes=2, pretrained=False, freeze_backbone=True)
        with pytest.raises(ValueError, match="num_layers must be positive"):
            model.unfreeze_backbone(num_layers=0)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"num_classes": 0}, "num_classes must be positive"),
            ({"dropout": 1.0}, "dropout must be in the range"),
        ],
    )
    def test_invalid_init_arguments(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            ResNetClassifier(pretrained=False, **kwargs)

    @pytest.mark.parametrize(
        ("model_name", "factory_name", "features"),
        [("resnet34", "resnet34", 512), ("resnet50", "resnet50", 2048)],
    )
    def test_supported_backbones_build_forwardable_classifier(
        self,
        monkeypatch,
        model_name,
        factory_name,
        features,
    ):
        monkeypatch.setattr(
            classifier_module.models,
            factory_name,
            lambda *, weights=None: _FakeResNetBackbone(features),
        )
        model = ResNetClassifier(
            num_classes=3,
            pretrained=False,
            model_name=model_name,
            freeze_backbone=False,
            dropout=0.25,
        )

        output = model(torch.randn(2, 3, 32, 32))

        assert output.shape == (2, 3)

    def test_unfreeze_backbone_without_num_layers_unfreezes_all_parameters(self):
        model = ResNetClassifier(num_classes=2, pretrained=False, freeze_backbone=True)

        backbone_flags_before = [
            param.requires_grad
            for name, param in model.backbone.named_parameters()
            if not name.startswith("fc.")
        ]
        assert backbone_flags_before
        assert all(flag is False for flag in backbone_flags_before)

        model.unfreeze_backbone()

        assert all(param.requires_grad for param in model.backbone.parameters())

    def test_predict_methods_temporarily_use_eval_mode_and_restore_state(self, monkeypatch):
        model = ResNetClassifier(num_classes=2, pretrained=False, freeze_backbone=False)
        model.train()
        observed_training_flags: list[bool] = []

        monkeypatch.setattr(
            model,
            "forward",
            lambda x: observed_training_flags.append(bool(model.training))
            or torch.tensor([[1.0, 0.0]]).repeat(x.shape[0], 1),
        )

        probabilities = model.predict_proba(torch.randn(2, 3, 64, 64))
        prediction = model.predict(torch.randn(2, 3, 64, 64))

        assert probabilities.shape == (2, 2)
        assert prediction.shape == (2,)
        assert observed_training_flags == [False, False]
        assert model.training is True


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

    def test_invalid_gamma(self):
        pred = torch.randn(4, 2)
        target = torch.randint(0, 2, (4,))
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            focal_loss(pred, target, gamma=-1.0)


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

    def test_requires_at_least_one_model(self):
        with pytest.raises(ValueError, match="at least one model"):
            EnsembleClassifier(models=[])
