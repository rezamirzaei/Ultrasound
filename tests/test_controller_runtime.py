"""Direct controller tests for pass-through and error-mapping behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from ultrasound.api.controllers.auth_controller import login, logout, me
from ultrasound.api.controllers.busi_yolo_controller import (
    busi_yolo_status,
    download_busi_yolo_model,
    get_busi_yolo_sample,
    predict_busi_yolo_sample,
)
from ultrasound.api.controllers.dashboard_controller import (
    get_busi_counts,
    get_busi_sample_preview,
    get_dashboard_readiness,
    get_dashboard_summary,
    get_industrial_sample_preview,
    get_industrial_segmentation_preview,
    get_industrial_summary,
    get_latest_busi_training,
    get_latest_industrial_training,
    get_ndt_sample,
    get_ndt_sample_signal,
    list_ndt_samples,
    run_busi_training,
    run_industrial_training,
)
from ultrasound.api.controllers.preprocessing_controller import preview_preprocessing
from ultrasound.api.controllers.yolo_training_controller import (
    liver_dataset_status,
    train_liver_yolo,
)
from ultrasound.api.models.domain import AuthSessionRecord
from ultrasound.api.models.schemas import (
    BusiTrainingRequest,
    IndustrialTrainingRequest,
    LoginRequest,
    PreprocessingRequest,
    YoloPredictRequest,
    YoloTrainRequest,
)
from ultrasound.api.services.service_errors import (
    InvalidRequestError,
    NotFoundError,
    UnauthorizedError,
)


def _raiser(exc: Exception):
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise exc

    return _raise


def _auth_session() -> AuthSessionRecord:
    return AuthSessionRecord(
        username="viewer",
        role="viewer",
        expires_at=datetime.now(tz=timezone.utc),
    )


def test_auth_login_me_and_logout_paths() -> None:
    session = _auth_session()
    auth_service = SimpleNamespace(
        authenticate=lambda username, password: session,
        issue_token=lambda current_session: "token-123",
        revoke_token=lambda token: True,
    )
    container = cast(Any, SimpleNamespace(auth_service=auth_service))

    login_response = login(LoginRequest(username="viewer", password="viewer123"), container=container)
    me_response = me(current_user=session)
    logout_response = logout(
        current_user=session,
        credentials=HTTPAuthorizationCredentials(scheme="Bearer", credentials="token-123"),
        container=container,
    )

    assert login_response.access_token == "token-123"
    assert me_response.username == "viewer"
    assert logout_response.success is True
    assert logout_response.revoked_token is True


def test_auth_login_maps_service_errors_and_logout_requires_bearer() -> None:
    failing_container = cast(
        Any,
        SimpleNamespace(auth_service=SimpleNamespace(authenticate=_raiser(UnauthorizedError("bad creds")))),
    )

    with pytest.raises(HTTPException) as exc:
        login(LoginRequest(username="viewer", password="bad1"), container=failing_container)
    assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED

    with pytest.raises(HTTPException) as logout_exc:
        logout(
            current_user=_auth_session(),
            credentials=None,
            container=cast(Any, SimpleNamespace(auth_service=SimpleNamespace())),
        )
    assert logout_exc.value.status_code == status.HTTP_401_UNAUTHORIZED


def test_dashboard_pass_through_routes_return_service_values() -> None:
    marker_summary = cast(Any, object())
    marker_readiness = cast(Any, object())
    marker_industrial_summary = cast(Any, object())
    marker_industrial_training = cast(Any, object())
    marker_busi_training = cast(Any, object())
    marker_ndt_list = cast(Any, object())
    marker_liver_status = cast(Any, object())
    marker_busi_yolo_status = cast(Any, object())
    marker_busi_model_status = cast(Any, object())
    marker_busi_sample = cast(Any, object())
    marker_busi_prediction = cast(Any, object())
    marker_preprocessing = cast(Any, object())

    container = cast(
        Any,
        SimpleNamespace(
            dashboard_service=SimpleNamespace(
                get_summary=lambda: marker_summary,
                get_data_readiness=lambda: marker_readiness,
                get_busi_counts=lambda: {"benign": 1},
                get_industrial_summary=lambda: marker_industrial_summary,
                list_ndt_samples=lambda: marker_ndt_list,
            ),
            busi_yolo_lab_service=SimpleNamespace(
                status=lambda: marker_busi_yolo_status,
                download_recommended_model=lambda force: marker_busi_model_status,
                get_sample=lambda class_name, sample_index: marker_busi_sample,
                predict=lambda class_name, sample_index, request: marker_busi_prediction,
            ),
            industrial_training_service=SimpleNamespace(
                get_latest_run=lambda dataset_name: marker_industrial_training
            ),
            busi_training_service=SimpleNamespace(
                get_latest_run=lambda include_normal: marker_busi_training
            ),
            liver_yolo_lab_service=SimpleNamespace(dataset_status=lambda: marker_liver_status),
            preprocessing_service=SimpleNamespace(preview=lambda request: marker_preprocessing),
        ),
    )

    assert get_dashboard_summary(container=container) is marker_summary
    assert get_dashboard_readiness(container=container) is marker_readiness
    assert get_busi_counts(container=container) == {"benign": 1}
    assert get_industrial_summary(container=container) is marker_industrial_summary
    assert get_latest_industrial_training(container=container) is marker_industrial_training
    assert get_latest_busi_training(container=container) is marker_busi_training
    assert list_ndt_samples(container=container) is marker_ndt_list
    assert liver_dataset_status(container=container) is marker_liver_status
    assert busi_yolo_status(container=container) is marker_busi_yolo_status
    assert download_busi_yolo_model(container=container) is marker_busi_model_status
    assert get_busi_yolo_sample("benign", 0, container=container) is marker_busi_sample
    assert (
        predict_busi_yolo_sample(
            "benign",
            0,
            YoloPredictRequest(model="yolov8n.pt"),
            container=container,
        )
        is marker_busi_prediction
    )
    assert preview_preprocessing(PreprocessingRequest(), container=container) is marker_preprocessing


def test_dashboard_and_training_routes_map_service_errors() -> None:
    container = cast(
        Any,
        SimpleNamespace(
            dashboard_service=SimpleNamespace(
                get_industrial_sample_preview=_raiser(NotFoundError("missing industrial sample")),
                get_busi_sample_preview=_raiser(NotFoundError("missing busi sample")),
                get_ndt_sample_detail=_raiser(NotFoundError("missing ndt sample")),
                get_ndt_signal_preview=_raiser(InvalidRequestError("bad max_points")),
            ),
            busi_yolo_lab_service=SimpleNamespace(
                download_recommended_model=_raiser(RuntimeError("offline")),
                get_sample=_raiser(NotFoundError("missing yolo sample")),
                predict=_raiser(InvalidRequestError("bad yolo request")),
            ),
            industrial_training_service=SimpleNamespace(
                run_training=_raiser(InvalidRequestError("bad industrial training request")),
                get_segmentation_preview=_raiser(NotFoundError("missing segmentation preview")),
            ),
            busi_training_service=SimpleNamespace(
                run_training=_raiser(InvalidRequestError("bad busi request"))
            ),
            liver_yolo_training_service=SimpleNamespace(
                train=_raiser(NotFoundError("missing liver dataset"))
            ),
            preprocessing_service=SimpleNamespace(preview=_raiser(InvalidRequestError("bad preprocessing"))),
        ),
    )

    with pytest.raises(HTTPException) as industrial_sample_exc:
        get_industrial_sample_preview("steel_defect", "train", "crazing", 0, container=container)
    assert industrial_sample_exc.value.status_code == status.HTTP_404_NOT_FOUND

    with pytest.raises(HTTPException) as industrial_training_exc:
        run_industrial_training(
            IndustrialTrainingRequest(dataset_name="steel_defect"),
            container=container,
        )
    assert industrial_training_exc.value.status_code == status.HTTP_400_BAD_REQUEST

    with pytest.raises(HTTPException) as segmentation_exc:
        get_industrial_segmentation_preview("steel_defect", "train", "crazing", 0, container=container)
    assert segmentation_exc.value.status_code == status.HTTP_404_NOT_FOUND

    with pytest.raises(HTTPException) as busi_training_exc:
        run_busi_training(BusiTrainingRequest(), container=container)
    assert busi_training_exc.value.status_code == status.HTTP_400_BAD_REQUEST

    with pytest.raises(HTTPException) as busi_sample_exc:
        get_busi_sample_preview("benign", 0, container=container)
    assert busi_sample_exc.value.status_code == status.HTTP_404_NOT_FOUND

    with pytest.raises(HTTPException) as ndt_sample_exc:
        get_ndt_sample("missing", container=container)
    assert ndt_sample_exc.value.status_code == status.HTTP_404_NOT_FOUND

    with pytest.raises(HTTPException) as ndt_signal_exc:
        get_ndt_sample_signal("missing", max_points=256, container=container)
    assert ndt_signal_exc.value.status_code == status.HTTP_400_BAD_REQUEST

    with pytest.raises(HTTPException) as liver_training_exc:
        train_liver_yolo(YoloTrainRequest(use_synthetic=False), container=container)
    assert liver_training_exc.value.status_code == status.HTTP_404_NOT_FOUND

    with pytest.raises(HTTPException) as download_exc:
        download_busi_yolo_model(container=container)
    assert download_exc.value.status_code == status.HTTP_502_BAD_GATEWAY

    with pytest.raises(HTTPException) as busi_yolo_sample_exc:
        get_busi_yolo_sample("benign", 0, container=container)
    assert busi_yolo_sample_exc.value.status_code == status.HTTP_404_NOT_FOUND

    with pytest.raises(HTTPException) as busi_yolo_predict_exc:
        predict_busi_yolo_sample(
            "benign",
            0,
            YoloPredictRequest(model="yolov8n.pt"),
            container=container,
        )
    assert busi_yolo_predict_exc.value.status_code == status.HTTP_400_BAD_REQUEST

    with pytest.raises(HTTPException) as preprocessing_exc:
        preview_preprocessing(PreprocessingRequest(), container=container)
    assert preprocessing_exc.value.status_code == status.HTTP_400_BAD_REQUEST
