"""Service layer for preprocessing preview workflows."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from ultrasound.api.models.schemas import (
    MethodMetrics,
    MethodPreview,
    PreprocessingPreviewResponse,
    PreprocessingRequest,
)
from ultrasound.api.services.interfaces import BusiSampleRepository, MediaRenderer
from ultrasound.api.services.service_errors import InvalidRequestError, NotFoundError
from ultrasound.preprocessing.denoising import admm_tv_denoising
from ultrasound.preprocessing.enhancement import ContrastEnhancer
from ultrasound.preprocessing.speckle import SpeckleReducer
from ultrasound.utils.metrics import compute_psnr, compute_rmse, compute_ssim


class PreprocessingService:
    """Orchestrates preprocessing algorithm runs for API consumers."""

    def __init__(self, dataset_repository: BusiSampleRepository, media_service: MediaRenderer):
        self.dataset_repository = dataset_repository
        self.media_service = media_service

    def _method_metrics(
        self,
        image: np.ndarray,
        reference: np.ndarray,
        cv: float,
    ) -> MethodMetrics:
        return MethodMetrics(
            rmse=float(compute_rmse(image, reference)),
            psnr=float(compute_psnr(image, reference, data_range=255.0)),
            ssim=float(compute_ssim(image, reference, data_range=255.0)),
            cv=float(cv),
        )

    def preview(self, request: PreprocessingRequest) -> PreprocessingPreviewResponse:
        try:
            sample = self.dataset_repository.get_busi_sample(
                class_name=request.class_name,
                index=request.sample_index,
            )
        except FileNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        image_rgb = sample.image_rgb
        gray = np.mean(image_rgb, axis=2).astype(np.uint8)

        lee = SpeckleReducer(method="lee", window_size=7)
        frost = SpeckleReducer(method="frost", window_size=5, damping_factor=1.5)
        enhancer = ContrastEnhancer(method="clahe", clip_limit=request.clip_limit)

        img_lee = lee.reduce(gray)
        img_frost = frost.reduce(gray)
        img_clahe = enhancer.enhance(gray)
        img_tv, _ = admm_tv_denoising(
            gray,
            lambda_tv=request.lambda_tv,
            rho=request.rho,
            n_iter=request.n_iter,
            verbose=False,
        )

        _, cv_lee = lee.estimate_speckle_level(img_lee)
        _, cv_frost = lee.estimate_speckle_level(img_frost)
        _, cv_clahe = lee.estimate_speckle_level(img_clahe)
        _, cv_tv = lee.estimate_speckle_level(img_tv)

        methods = [
            MethodPreview(
                name="Lee Filter",
                image_data_url=self.media_service.as_png_data_url(img_lee),
                metrics=self._method_metrics(img_lee, gray, cv_lee),
            ),
            MethodPreview(
                name="Frost Filter",
                image_data_url=self.media_service.as_png_data_url(img_frost),
                metrics=self._method_metrics(img_frost, gray, cv_frost),
            ),
            MethodPreview(
                name="CLAHE",
                image_data_url=self.media_service.as_png_data_url(img_clahe),
                metrics=self._method_metrics(img_clahe, gray, cv_clahe),
            ),
            MethodPreview(
                name="ADMM-TV",
                image_data_url=self.media_service.as_png_data_url(img_tv),
                metrics=self._method_metrics(img_tv, gray, cv_tv),
            ),
        ]

        recommendation = max(methods, key=lambda item: item.metrics.ssim).name

        return PreprocessingPreviewResponse(
            image_shape=[int(v) for v in image_rgb.shape],
            original_image_data_url=self.media_service.as_png_data_url(gray),
            methods=methods,
            recommendation=recommendation,
            generated_at=datetime.now(tz=timezone.utc),
        )
