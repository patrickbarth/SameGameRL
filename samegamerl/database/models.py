"""SQLAlchemy ORM models for SameGameRL database."""

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class GameConfig(Base):
    __tablename__ = "game_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    num_rows = Column(Integer, nullable=False)
    num_cols = Column(Integer, nullable=False)
    num_colors = Column(Integer, nullable=False)
    name = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    benchmark_sets = relationship("BenchmarkSet", back_populates="config")

    # Constraints
    __table_args__ = (
        UniqueConstraint("num_rows", "num_cols", "num_colors"),
    )

    def __repr__(self) -> str:
        return f"<GameConfig({self.num_rows}x{self.num_cols}, {self.num_colors} colors)>"


class BenchmarkSet(Base):
    __tablename__ = "benchmark_sets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    config_id = Column(Integer, ForeignKey("game_configs.id"), nullable=False)
    num_games = Column(Integer, nullable=False)
    base_seed = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    config = relationship("GameConfig", back_populates="benchmark_sets")
    games = relationship("Game", back_populates="benchmark_set", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        UniqueConstraint("config_id", "base_seed", "num_games"),
    )

    def __repr__(self) -> str:
        return f"<BenchmarkSet({self.num_games} games, seed={self.base_seed})>"


class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_set_id = Column(Integer, ForeignKey("benchmark_sets.id"), nullable=False)
    game_index = Column(Integer, nullable=False)
    board_state = Column(JSON, nullable=False)
    seed = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    benchmark_set = relationship("BenchmarkSet", back_populates="games")
    results = relationship("GameResult", back_populates="game", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        UniqueConstraint("benchmark_set_id", "game_index"),
    )

    def __repr__(self) -> str:
        return f"<Game(set={self.benchmark_set_id}, index={self.game_index})>"


class Bot(Base):
    __tablename__ = "bots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)
    bot_type = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    results = relationship("GameResult", back_populates="bot")

    def __repr__(self) -> str:
        return f"<Bot({self.name}, type={self.bot_type})>"


class GameResult(Base):
    __tablename__ = "game_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    bot_id = Column(Integer, ForeignKey("bots.id"), nullable=False)
    tiles_cleared = Column(Integer, nullable=False)
    singles_remaining = Column(Integer, nullable=False)
    moves_made = Column(Integer, nullable=False)
    completed = Column(Boolean, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    game = relationship("Game", back_populates="results")
    bot = relationship("Bot", back_populates="results")

    # Constraints
    __table_args__ = (
        UniqueConstraint("game_id", "bot_id"),
    )

    def __repr__(self) -> str:
        return f"<GameResult(game={self.game_id}, bot={self.bot_id}, cleared={self.tiles_cleared})>"