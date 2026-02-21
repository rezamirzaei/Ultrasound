"""Dependency container for API components."""

from __future__ import annotations

from ultrasound.api.config import AppConfig
from ultrasound.api.repositories.dataset_repository import DatasetRepository
from ultrasound.api.services.auth_service import AuthService
from ultrasound.api.services.busi_training_service import BusiTrainingService
from ultrasound.api.services.dashboard_service import DashboardService
from ultrasound.api.services.error_analytics_service import ErrorAnalyticsService
from ultrasound.api.services.media_service import MediaService
from ultrasound.api.services.ndt_detection_service import NdtDetectionService
from ultrasound.api.services.preprocessing_service import PreprocessingService


class ApplicationContainer:
    """Holds long-lived service objects used by controllers."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig.from_project_root()

        self.dataset_repository = DatasetRepository(self.config)
        self.auth_service = AuthService()
        self.error_analytics_service = ErrorAnalyticsService()
        self.media_service = MediaService()
        self.ndt_detection_service = NdtDetectionService()
        self.dashboard_service = DashboardService(
            self.dataset_repository,
            self.media_service,
            self.ndt_detection_service,
        )
        self.busi_training_service = BusiTrainingService(self.dataset_repository)
        self.preprocessing_service = PreprocessingService(
            self.dataset_repository, self.media_service
        )
