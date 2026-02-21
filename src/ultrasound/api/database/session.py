"""SQLAlchemy engine/session setup."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()


class DatabaseSessionManager:
    """Owns SQLAlchemy engine/session factory and schema lifecycle."""

    def __init__(self, database_url: str):
        connect_args: dict[str, object] = {}
        if database_url.startswith("sqlite:///"):
            connect_args["check_same_thread"] = False

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
