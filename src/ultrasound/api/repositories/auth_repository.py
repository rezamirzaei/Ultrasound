"""Repository layer for authentication users and tokens."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

from sqlalchemy import delete, select
from sqlalchemy.engine import CursorResult

from ultrasound.api.database.models import AuthTokenORM, AuthUserORM
from ultrasound.api.database.session import DatabaseSessionManager


class AuthRepository:
    """Encapsulates auth-related persistence via SQLAlchemy ORM."""

    def __init__(self, db: DatabaseSessionManager):
        self.db = db
        self.db.create_schema()

    def get_user_by_username(self, username: str) -> AuthUserORM | None:
        with self.db.session_scope() as session:
            return session.scalars(
                select(AuthUserORM).where(AuthUserORM.username == username).limit(1)
            ).first()

    def list_users(self) -> list[AuthUserORM]:
        with self.db.session_scope() as session:
            return list(session.scalars(select(AuthUserORM).order_by(AuthUserORM.username)).all())

    def create_or_update_user(
        self,
        username: str,
        role: str,
        password_hash: str,
        is_active: bool = True,
        force_password_update: bool = False,
    ) -> AuthUserORM:
        with self.db.session_scope() as session:
            user = session.scalars(
                select(AuthUserORM).where(AuthUserORM.username == username).limit(1)
            ).first()
            if user is None:
                user = AuthUserORM(
                    username=username,
                    role=role,
                    password_hash=password_hash,
                    is_active=bool(is_active),
                )
                session.add(user)
                session.flush()
                return user

            user.role = role
            user.is_active = bool(is_active)
            if force_password_update:
                user.password_hash = password_hash
            session.flush()
            return user

    def issue_token(self, user_id: int, token_hash: str, expires_at: datetime) -> int:
        with self.db.session_scope() as session:
            token = AuthTokenORM(
                user_id=int(user_id),
                token_hash=token_hash,
                expires_at=expires_at,
            )
            session.add(token)
            session.flush()
            if token.id is None:
                raise RuntimeError("Failed to persist auth token")
            return int(token.id)

    def get_token(self, token_hash: str) -> AuthTokenORM | None:
        with self.db.session_scope() as session:
            return session.scalars(
                select(AuthTokenORM).where(AuthTokenORM.token_hash == token_hash).limit(1)
            ).first()

    def get_token_with_user(self, token_hash: str) -> tuple[AuthTokenORM, AuthUserORM] | None:
        with self.db.session_scope() as session:
            row = session.execute(
                select(AuthTokenORM, AuthUserORM)
                .join(AuthUserORM, AuthUserORM.id == AuthTokenORM.user_id)
                .where(AuthTokenORM.token_hash == token_hash)
                .limit(1)
            ).first()
            if row is None:
                return None
            token, user = row
            return token, user

    def revoke_token(self, token_hash: str) -> bool:
        with self.db.session_scope() as session:
            token = session.scalars(
                select(AuthTokenORM).where(AuthTokenORM.token_hash == token_hash).limit(1)
            ).first()
            if token is None:
                return False
            token.revoked_at = datetime.now(tz=timezone.utc)
            session.flush()
            return True

    def touch_token(self, token_id: int) -> None:
        with self.db.session_scope() as session:
            token = session.get(AuthTokenORM, int(token_id))
            if token is None:
                return
            token.last_used_at = datetime.now(tz=timezone.utc)
            session.flush()

    def purge_expired_tokens(self) -> int:
        now = datetime.now(tz=timezone.utc)
        with self.db.session_scope() as session:
            stale_ids = session.scalars(
                select(AuthTokenORM.id).where(AuthTokenORM.expires_at < now)
            ).all()
            if not stale_ids:
                return 0
            result = cast(
                CursorResult[Any],
                session.execute(delete(AuthTokenORM).where(AuthTokenORM.id.in_(stale_ids))),
            )
            return int(result.rowcount or 0)
