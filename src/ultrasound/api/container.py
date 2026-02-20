"""Dependency container for API components."""

from __future__ import annotations

from ultrasound.api.config import AppConfig
from ultrasound.api.repositories.dataset_repository import DatasetRepository
from ultrasound.api.services.dashboard_service import DashboardService
from ultrasound.api.services.preprocessing_service import PreprocessingService


class ApplicationContainer:
    """Holds long-lived service objects used by controllers."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig.from_project_root()

        self.dataset_repository = DatasetRepository(self.config)
        self.dashboard_service = DashboardService(self.dataset_repository)
        self.preprocessing_service = PreprocessingService(self.dataset_repository)
