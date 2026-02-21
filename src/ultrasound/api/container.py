"""Dependency container for API components."""

from __future__ import annotations

from ultrasound.api.config import AppConfig
from ultrasound.api.database.session import DatabaseSessionManager
from ultrasound.api.repositories.auth_repository import AuthRepository
from ultrasound.api.repositories.dataset_repository import DatasetRepository
from ultrasound.api.repositories.job_repository import JobRepository
from ultrasound.api.services.auth_service import AuthService
from ultrasound.api.services.busi_training_service import BusiTrainingService
from ultrasound.api.services.dashboard_service import DashboardService
from ultrasound.api.services.data_ingestion_service import DataIngestionService
from ultrasound.api.services.dataset_upload_service import DatasetUploadService
from ultrasound.api.services.error_analytics_service import ErrorAnalyticsService
from ultrasound.api.services.industrial_training_service import IndustrialTrainingService
from ultrasound.api.services.job_queue_service import JobQueueService
from ultrasound.api.services.media_service import MediaService
from ultrasound.api.services.ndt_detection_service import NdtDetectionService
from ultrasound.api.services.observability_service import ObservabilityService
from ultrasound.api.services.preprocessing_service import PreprocessingService


class ApplicationContainer:
    """Holds long-lived service objects used by controllers."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig.from_project_root()
        database_url = self.config.database_url or (
            f"sqlite:///{(self.config.data_dir / 'inphase.sqlite3').resolve()}"
        )
        self.db = DatabaseSessionManager(database_url)

        self.auth_repository = AuthRepository(self.db)
        self.dataset_repository = DatasetRepository(self.config, self.db)
        self.job_repository = JobRepository(self.db)

        self.observability_service = ObservabilityService()
        self.auth_service = AuthService(self.auth_repository)
        self.error_analytics_service = ErrorAnalyticsService(self.db)
        self.media_service = MediaService()
        self.ndt_detection_service = NdtDetectionService()
        self.dashboard_service = DashboardService(
            self.dataset_repository,
            self.media_service,
            self.ndt_detection_service,
        )
        self.data_ingestion_service = DataIngestionService(self.dataset_repository)
        self.busi_training_service = BusiTrainingService(self.dataset_repository)
        self.industrial_training_service = IndustrialTrainingService(
            self.dataset_repository,
            self.media_service,
        )
        self.dataset_upload_service = DatasetUploadService(self.dataset_repository)
        self.preprocessing_service = PreprocessingService(
            self.dataset_repository, self.media_service
        )
        self.job_queue_service = JobQueueService(
            repository=self.job_repository,
            busi_training_service=self.busi_training_service,
            industrial_training_service=self.industrial_training_service,
            data_ingestion_service=self.data_ingestion_service,
            observability_service=self.observability_service,
        )
