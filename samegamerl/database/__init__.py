"""Database package for SameGameRL benchmarking system."""

# Lazy imports to avoid requiring database dependencies unless needed
from samegamerl.database.availability import DATABASE_AVAILABLE

if DATABASE_AVAILABLE:
    from samegamerl.database.connection import DatabaseManager, db_manager
    from samegamerl.database.models import (
        Base,
        Bot,
        GamePool,
        Game,
        GameConfig,
        GameResult,
    )
    from samegamerl.database.repository import DatabaseRepository

    __all__ = [
        "DatabaseManager",
        "db_manager",
        "DatabaseRepository",
        "Base",
        "GameConfig",
        "GamePool",
        "Game",
        "Bot",
        "GameResult",
        "DATABASE_AVAILABLE",
    ]
else:
    __all__ = ["DATABASE_AVAILABLE"]