"""Alembic migration bootstrap — run `alembic upgrade head` after setup."""

from backend.database.base import Base
from backend.database.session import engine

# Import models so metadata is populated
import backend.database.models  # noqa: F401

target_metadata = Base.metadata
