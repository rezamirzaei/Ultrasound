"""SQLAlchemy engine/session setup."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class DatabaseSessionManager:
    """Owns SQLAlchemy engine/session factory and schema lifecycle."""

    def __init__(self, database_url: str):
        connect_args: dict[str, object] = {}
        if self._is_sqlite_url(database_url):
            connect_args["check_same_thread"] = False
            self._ensure_sqlite_parent_dir(database_url)

        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        self._session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            autoflush=False,
            class_=Session,
        )

    def create_schema(self) -> None:
        """Create tables if they do not exist."""
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional session context."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _is_sqlite_url(database_url: str) -> bool:
        try:
            return make_url(database_url).get_backend_name() == "sqlite"
        except Exception:
            return database_url.startswith("sqlite:///")

    @staticmethod
    def _ensure_sqlite_parent_dir(database_url: str) -> None:
        try:
            database_path = make_url(database_url).database
        except Exception:
            return
        if not database_path or database_path == ":memory:":
            return
        Path(database_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
