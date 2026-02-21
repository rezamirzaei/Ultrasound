"""API controller routers."""

from .auth_controller import router as auth_router
from .dashboard_controller import router as dashboard_router
from .health_controller import router as health_router
from .mlops_controller import router as mlops_router
from .ops_controller import router as ops_router
from .preprocessing_controller import router as preprocessing_router

__all__ = [
    "auth_router",
    "dashboard_router",
    "health_router",
    "mlops_router",
    "ops_router",
    "preprocessing_router",
]
