"""API controller routers."""

from .auth_controller import router as auth_router
from .busi_yolo_controller import router as busi_yolo_router
from .dashboard_controller import router as dashboard_router
from .health_controller import router as health_router
from .mlops_controller import router as mlops_router
from .ops_controller import router as ops_router
from .preprocessing_controller import router as preprocessing_router
from .yolo_controller import router as yolo_router
from .yolo_training_controller import router as yolo_training_router

__all__ = [
    "auth_router",
    "busi_yolo_router",
    "dashboard_router",
    "health_router",
    "mlops_router",
    "ops_router",
    "preprocessing_router",
    "yolo_router",
    "yolo_training_router",
]
