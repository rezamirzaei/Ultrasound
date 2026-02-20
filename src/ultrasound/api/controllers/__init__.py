"""API controller routers."""

from .dashboard_controller import router as dashboard_router
from .health_controller import router as health_router
from .preprocessing_controller import router as preprocessing_router

__all__ = ["dashboard_router", "health_router", "preprocessing_router"]
