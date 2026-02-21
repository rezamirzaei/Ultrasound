"""Token-based authentication and role resolution."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta, timezone

from ultrasound.api.models.domain import AuthSessionRecord


class AuthService:
    """Issues and validates signed bearer tokens for API requests."""

    ROLE_ORDER = {"viewer": 1, "analyst": 2, "admin": 3}

    def __init__(self) -> None:
        self.secret = os.getenv("INPHASE_AUTH_SECRET", "inphase-dev-secret-change-me").encode(
            "utf-8"
        )
        self.token_ttl_minutes = max(5, int(os.getenv("INPHASE_TOKEN_TTL_MINUTES", "480")))

        self._users: dict[str, tuple[str, str]] = {
            "viewer": (os.getenv("INPHASE_VIEWER_PASSWORD", "viewer123"), "viewer"),
            "analyst": (os.getenv("INPHASE_ANALYST_PASSWORD", "analyst123"), "analyst"),
            "admin": (os.getenv("INPHASE_ADMIN_PASSWORD", "admin123"), "admin"),
        }

    def authenticate(self, username: str, password: str) -> AuthSessionRecord:
        normalized = username.strip().lower()
        entry = self._users.get(normalized)
        if entry is None:
            raise ValueError("Invalid credentials")

        expected_password, role = entry
        if not hmac.compare_digest(expected_password, password):
            raise ValueError("Invalid credentials")

        return AuthSessionRecord(
            username=normalized,
            role=role,  # type: ignore[arg-type]
            expires_at=datetime.now(tz=timezone.utc) + timedelta(minutes=self.token_ttl_minutes),
        )

    def issue_token(self, session: AuthSessionRecord) -> str:
        payload = {
            "sub": session.username,
            "role": session.role,
            "exp": int(session.expires_at.timestamp()),
            "iat": int(datetime.now(tz=timezone.utc).timestamp()),
        }
        payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        payload_b64 = self._b64url_encode(payload_bytes)
        signature = self._sign(payload_b64.encode("ascii"))
        return f"{payload_b64}.{self._b64url_encode(signature)}"

    def verify_token(self, token: str) -> AuthSessionRecord:
        token = token.strip()
        if "." not in token:
            raise ValueError("Invalid token format")

        payload_b64, signature_b64 = token.split(".", 1)
        expected_signature = self._sign(payload_b64.encode("ascii"))
        provided_signature = self._b64url_decode(signature_b64)
        if not hmac.compare_digest(expected_signature, provided_signature):
            raise ValueError("Invalid token signature")

        payload = json.loads(self._b64url_decode(payload_b64).decode("utf-8"))
        username = str(payload.get("sub", "")).strip().lower()
        role = str(payload.get("role", "")).strip().lower()
        exp_ts = int(payload.get("exp", 0))
        if not username or role not in self.ROLE_ORDER:
            raise ValueError("Invalid token payload")

        expires_at = datetime.fromtimestamp(exp_ts, tz=timezone.utc)
        if datetime.now(tz=timezone.utc) >= expires_at:
            raise ValueError("Token expired")

        return AuthSessionRecord(
            username=username,
            role=role,  # type: ignore[arg-type]
            expires_at=expires_at,
        )

    def has_role(self, actual_role: str, required_role: str) -> bool:
        actual_rank = self.ROLE_ORDER.get(actual_role, 0)
        required_rank = self.ROLE_ORDER.get(required_role, 10)
        return actual_rank >= required_rank

    def _sign(self, payload: bytes) -> bytes:
        return hmac.new(self.secret, payload, hashlib.sha256).digest()

    def _b64url_encode(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")

    def _b64url_decode(self, value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(value + padding)
