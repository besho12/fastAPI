from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import settings
from app.database.connection import Base

# IMPORTANT:
# Import models so SQLAlchemy registers all ORM tables
# in Base.metadata before Alembic autogenerate runs.
import app.database.models  # noqa: F401


# --------------------------------------------------------------------------
# Alembic Config
# --------------------------------------------------------------------------

config = context.config


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# --------------------------------------------------------------------------
# Database URL
# --------------------------------------------------------------------------

database_url = getattr(settings, "DATABASE_URL", None)

if not database_url:
    raise RuntimeError(
        "DATABASE_URL is not configured. "
        "Set DATABASE_URL in the .env file."
    )

config.set_main_option(
    "sqlalchemy.url",
    database_url.replace("%", "%%"),
)


# --------------------------------------------------------------------------
# SQLAlchemy metadata
# --------------------------------------------------------------------------

target_metadata = Base.metadata


# --------------------------------------------------------------------------
# Offline migrations
# --------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """
    Run migrations without establishing a live database connection.
    """

    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named",
        },
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# --------------------------------------------------------------------------
# Online migrations
# --------------------------------------------------------------------------

def run_migrations_online() -> None:
    """
    Run migrations using a live PostgreSQL connection.
    """

    connectable = engine_from_config(
        config.get_section(
            config.config_ini_section,
            {},
        ),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()