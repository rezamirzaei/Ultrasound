"""Tests for authentication repository persistence behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from ultrasound.api.database.session import DatabaseSessionManager
from ultrasound.api.repositories.auth_repository import AuthRepository


def _make_repository(tmp_path: Path) -> AuthRepository:
    db = DatabaseSessionManager(f"sqlite:///{(tmp_path / 'auth.sqlite3').resolve()}")
    return AuthRepository(db)


def test_create_update_list_and_lookup_users(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)

    created = repository.create_or_update_user(
        username="viewer",
        role="viewer",
        password_hash="hash-1",
        is_active=True,
    )

    assert created.id is not None
    assert repository.get_user_by_username("viewer") is not None
    assert [user.username for user in repository.list_users()] == ["viewer"]

    unchanged_password = repository.create_or_update_user(
        username="viewer",
        role="analyst",
        password_hash="hash-2",
        is_active=False,
        force_password_update=False,
    )
    assert unchanged_password.role == "analyst"
    assert unchanged_password.password_hash == "hash-1"
    assert unchanged_password.is_active is False

    updated_password = repository.create_or_update_user(
        username="viewer",
        role="admin",
        password_hash="hash-3",
        is_active=True,
        force_password_update=True,
    )
    assert updated_password.role == "admin"
    assert updated_password.password_hash == "hash-3"
    assert updated_password.is_active is True


def test_token_lifecycle_operations(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)
    user = repository.create_or_update_user(
        username="viewer",
        role="viewer",
        password_hash="hash-1",
        is_active=True,
    )
    assert user.id is not None

    future_expiry = datetime.now(tz=timezone.utc) + timedelta(minutes=30)
    token_id = repository.issue_token(int(user.id), "active-token", future_expiry)

    token = repository.get_token("active-token")
    assert token is not None
    assert token.id == token_id

    lookup = repository.get_token_with_user("active-token")
    assert lookup is not None
    token_row, token_user = lookup
    assert token_row.token_hash == "active-token"
    assert token_user.username == "viewer"

    repository.touch_token(token_id)
    touched = repository.get_token("active-token")
    assert touched is not None
    assert touched.last_used_at is not None

    assert repository.revoke_token("active-token") is True
    revoked = repository.get_token("active-token")
    assert revoked is not None
    assert revoked.revoked_at is not None

    assert repository.revoke_token("missing-token") is False
    assert repository.get_token("missing-token") is None
    assert repository.get_token_with_user("missing-token") is None


def test_purge_expired_tokens_removes_only_stale_rows(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)
    user = repository.create_or_update_user(
        username="viewer",
        role="viewer",
        password_hash="hash-1",
        is_active=True,
    )
    assert user.id is not None

    past_expiry = datetime.now(tz=timezone.utc) - timedelta(minutes=5)
    future_expiry = datetime.now(tz=timezone.utc) + timedelta(minutes=5)
    repository.issue_token(int(user.id), "expired-token", past_expiry)
    repository.issue_token(int(user.id), "active-token", future_expiry)

    assert repository.purge_expired_tokens() == 1
    assert repository.get_token("expired-token") is None
    assert repository.get_token("active-token") is not None
    assert repository.purge_expired_tokens() == 0
