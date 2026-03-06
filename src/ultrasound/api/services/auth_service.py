"""Database-backed authentication and token session management."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Literal, cast

from ultrasound.api.models.domain import AuthSessionRecord
from ultrasound.api.repositories.auth_repository import AuthRepository
from ultrasound.api.services.service_errors import InvalidRequestError, UnauthorizedError


class AuthService:
    """Issues and validates bearer tokens persisted in the database."""

    ROLE_ORDER = {"viewer": 1, "analyst": 2, "admin": 3}
    PASSWORD_SCHEME = "pbkdf2_sha256"

    def __init__(self, repository: AuthRepository) -> None:
        self.repository = repository
        self.token_ttl_minutes = max(5, int(os.getenv("INPHASE_TOKEN_TTL_MINUTES", "480")))
        self.password_iterations = max(
            120_000, int(os.getenv("INPHASE_PBKDF2_ITERATIONS", "260000"))
        )
        self.password_salt_bytes = max(16, int(os.getenv("INPHASE_PBKDF2_SALT_BYTES", "16")))
        self.force_default_users = os.getenv("INPHASE_FORCE_DEFAULT_USERS", "0").strip() in {
            "1",
            "true",
            "yes",
        }
        self._bootstrap_default_users()

    def _normalize_role(self, value: str) -> Literal["viewer", "analyst", "admin"]:
        role = value.strip().lower()
        if role not in self.ROLE_ORDER:
            raise InvalidRequestError("Invalid role")
        return cast(Literal["viewer", "analyst", "admin"], role)

    def _hash_password(self, password: str) -> str:
        salt = secrets.token_bytes(self.password_salt_bytes)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self.password_iterations,
        )
        return f"{self.PASSWORD_SCHEME}${self.password_iterations}$" f"{salt.hex()}${digest.hex()}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        try:
            scheme, iterations_raw, salt_hex, digest_hex = password_hash.split("$", 3)
            iterations = int(iterations_raw)
            salt = bytes.fromhex(salt_hex)
            expected_digest = bytes.fromhex(digest_hex)
        except Exception:
            return False

        if scheme != self.PASSWORD_SCHEME:
            return False

        computed = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(computed, expected_digest)

    def _token_hash(self, token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _to_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _bootstrap_default_users(self) -> None:
        defaults = [
            ("viewer", os.getenv("INPHASE_VIEWER_PASSWORD", "viewer123"), "viewer"),
            ("analyst", os.getenv("INPHASE_ANALYST_PASSWORD", "analyst123"), "analyst"),
            ("admin", os.getenv("INPHASE_ADMIN_PASSWORD", "admin123"), "admin"),
        ]
        for username, password, role in defaults:
            self.repository.create_or_update_user(
                username=username.strip().lower(),
                role=self._normalize_role(role),
                password_hash=self._hash_password(password),
                is_active=True,
                force_password_update=self.force_default_users,
            )
        self.repository.purge_expired_tokens()

    def authenticate(self, username: str, password: str) -> AuthSessionRecord:
        normalized = username.strip().lower()
        user = self.repository.get_user_by_username(normalized)
        if user is None or not bool(user.is_active):
            raise UnauthorizedError("Invalid credentials")
        if not self._verify_password(password, str(user.password_hash)):
            raise UnauthorizedError("Invalid credentials")

        expires_at = datetime.now(tz=timezone.utc) + timedelta(minutes=self.token_ttl_minutes)
        return AuthSessionRecord(
            username=normalized,
            role=self._normalize_role(str(user.role)),
            expires_at=expires_at,
        )

    def issue_token(self, session: AuthSessionRecord) -> str:
        user = self.repository.get_user_by_username(session.username.strip().lower())
        if user is None or not bool(user.is_active):
            raise UnauthorizedError("Invalid user")
        if user.id is None:
            raise UnauthorizedError("Invalid user id")

        token = secrets.token_urlsafe(48)
        self.repository.issue_token(
            user_id=int(user.id),
            token_hash=self._token_hash(token),
            expires_at=self._to_utc(session.expires_at),
        )
        self.repository.purge_expired_tokens()
        return token

    def verify_token(self, token: str) -> AuthSessionRecord:
        token = token.strip()
        if not token:
            raise UnauthorizedError("Invalid token")

        lookup = self.repository.get_token_with_user(self._token_hash(token))
        if lookup is None:
            raise UnauthorizedError("Invalid token")

        token_row, user = lookup
        now = datetime.now(tz=timezone.utc)

        expires_at = self._to_utc(token_row.expires_at)
        if token_row.revoked_at is not None:
            raise UnauthorizedError("Token revoked")
        if expires_at <= now:
            raise UnauthorizedError("Token expired")
        if not bool(user.is_active):
            raise UnauthorizedError("User is inactive")
        if user.id is None or token_row.id is None:
            raise UnauthorizedError("Invalid token state")

        self.repository.touch_token(int(token_row.id))
        return AuthSessionRecord(
            username=str(user.username).strip().lower(),
            role=self._normalize_role(str(user.role)),
            expires_at=expires_at,
        )

    def revoke_token(self, token: str) -> bool:
        token = token.strip()
        if not token:
            return False
        return self.repository.revoke_token(self._token_hash(token))

    def has_role(self, actual_role: str, required_role: str) -> bool:
        actual_rank = self.ROLE_ORDER.get(actual_role, 0)
        required_rank = self.ROLE_ORDER.get(required_role, 10)
        return actual_rank >= required_rank
