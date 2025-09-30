"""Simple database connection management."""

import os
from dataclasses import dataclass

import asyncpg
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load environment variables from .env file
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "samegamerl_dev"
    username: str = "postgres"
    password: str = ""

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables or .env file."""
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "samegamerl_dev"),
            username=os.getenv("DB_USER", os.getenv("USER", "postgres")),
            password=os.getenv("DB_PASSWORD", ""),
        )


class DatabaseManager:
    """Simple database connection manager."""

    def __init__(self, config: DatabaseConfig | None = None):
        self.config = config or DatabaseConfig.from_env()
        self._engine = None
        self._session_maker = None

    @property
    def sync_url(self) -> str:
        """SQLAlchemy connection URL."""
        password_part = f":{self.config.password}" if self.config.password else ""
        return f"postgresql://{self.config.username}{password_part}@{self.config.host}:{self.config.port}/{self.config.database}"

    @property
    def async_url(self) -> str:
        """asyncpg connection URL."""
        password_part = f":{self.config.password}" if self.config.password else ""
        return f"postgresql://{self.config.username}{password_part}@{self.config.host}:{self.config.port}/{self.config.database}"

    def get_engine(self):
        """Get SQLAlchemy engine (creates if needed)."""
        if self._engine is None:
            self._engine = create_engine(self.sync_url)
        return self._engine

    def get_session(self):
        """Get SQLAlchemy session."""
        if self._session_maker is None:
            self._session_maker = sessionmaker(bind=self.get_engine())
        return self._session_maker()

    async def get_async_connection(self):
        """Get asyncpg connection."""
        return await asyncpg.connect(self.async_url)

    async def test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            conn = await self.get_async_connection()
            result = await conn.fetchval("SELECT 1")
            await conn.close()
            return result == 1
        except Exception:
            return False


# Global instance for easy access
db_manager = DatabaseManager()