"""Repository pattern for database operations."""

from typing import Any, Optional

from sqlalchemy.orm import Session

from samegamerl.database.connection import db_manager
from samegamerl.database.models import Bot, GamePool, Game, GameConfig, GameResult


class GameConfigRepository:
    def __init__(self, session: Session):
        self.session = session

    def find_or_create(self, num_rows: int, num_cols: int, num_colors: int, name: str = None) -> GameConfig:
        """Find existing config or create new one."""
        config = (
            self.session.query(GameConfig)
            .filter_by(num_rows=num_rows, num_cols=num_cols, num_colors=num_colors)
            .first()
        )

        if not config:
            config = GameConfig(
                num_rows=num_rows,
                num_cols=num_cols,
                num_colors=num_colors,
                name=name
            )
            self.session.add(config)
            self.session.flush()  # Get ID without committing

        return config

    def get_by_id(self, config_id: int) -> Optional[GameConfig]:
        """Get config by ID."""
        return self.session.query(GameConfig).filter_by(id=config_id).first()


class GamePoolRepository:
    def __init__(self, session: Session):
        self.session = session

    def find_or_create(self, config: GameConfig, base_seed: int, max_games: int) -> GamePool:
        """Find existing game pool or create new one."""
        pool = (
            self.session.query(GamePool)
            .filter_by(config_id=config.id, base_seed=base_seed)
            .first()
        )

        if not pool:
            pool = GamePool(
                config_id=config.id,
                base_seed=base_seed,
                max_games=max_games
            )
            self.session.add(pool)
            self.session.flush()
        elif pool.max_games < max_games:
            # Extend existing pool if we need more games
            pool.max_games = max_games
            self.session.flush()

        return pool

    def get_by_id(self, pool_id: int) -> Optional[GamePool]:
        """Get game pool by ID."""
        return self.session.query(GamePool).filter_by(id=pool_id).first()

    def get_by_config_and_seed(self, config_id: int, base_seed: int) -> Optional[GamePool]:
        """Get game pool by config and seed."""
        return (
            self.session.query(GamePool)
            .filter_by(config_id=config_id, base_seed=base_seed)
            .first()
        )


class GameRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_game(self, pool: GamePool, game_index: int,
                   board_state: list[list[int]], seed: int) -> Game:
        """Create a new game in a pool."""
        game = Game(
            pool_id=pool.id,
            game_index=game_index,
            board_state=board_state,  # SQLAlchemy will JSON-serialize this
            seed=seed
        )
        self.session.add(game)
        self.session.flush()
        return game

    def get_games_for_pool(self, pool_id: int, limit: int = None) -> list[Game]:
        """Get games for a pool, optionally limited to first N games."""
        query = (
            self.session.query(Game)
            .filter_by(pool_id=pool_id)
            .order_by(Game.game_index)
        )

        if limit is not None:
            query = query.limit(limit)

        return query.all()

    def get_benchmark_games(self, config_id: int, base_seed: int, num_games: int) -> list[Game]:
        """Get N games for a benchmark (efficient reuse pattern)."""
        pool = (
            self.session.query(GamePool)
            .filter_by(config_id=config_id, base_seed=base_seed)
            .first()
        )

        if not pool:
            return []

        return self.get_games_for_pool(pool.id, limit=num_games)


class BotRepository:
    def __init__(self, session: Session):
        self.session = session

    def find_or_create(self, name: str, bot_type: str) -> Bot:
        """Find existing bot or create new one."""
        bot = self.session.query(Bot).filter_by(name=name).first()

        if not bot:
            bot = Bot(name=name, bot_type=bot_type)
            self.session.add(bot)
            self.session.flush()

        return bot

    def get_by_name(self, name: str) -> Optional[Bot]:
        """Get bot by name."""
        return self.session.query(Bot).filter_by(name=name).first()


class GameResultRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_result(self, game: Game, bot: Bot, tiles_cleared: int,
                     singles_remaining: int, moves_made: int, completed: bool) -> GameResult:
        """Create a game result."""
        result = GameResult(
            game_id=game.id,
            bot_id=bot.id,
            tiles_cleared=tiles_cleared,
            singles_remaining=singles_remaining,
            moves_made=moves_made,
            completed=completed
        )
        self.session.add(result)
        self.session.flush()
        return result

    def get_results_for_bot(self, bot_id: int) -> list[GameResult]:
        """Get all results for a specific bot."""
        return self.session.query(GameResult).filter_by(bot_id=bot_id).all()

    def get_results_for_pool(self, pool_id: int, bot_id: int = None, limit: int = None) -> list[GameResult]:
        """Get results for a game pool, optionally filtered by bot and limited to N games."""
        query = (
            self.session.query(GameResult)
            .join(Game)
            .filter(Game.pool_id == pool_id)
            .order_by(Game.game_index)
        )

        if bot_id:
            query = query.filter(GameResult.bot_id == bot_id)

        if limit:
            query = query.limit(limit)

        return query.all()

    def get_benchmark_results(self, config_id: int, base_seed: int, num_games: int, bot_id: int = None) -> list[GameResult]:
        """Get results for a benchmark (first N games), optionally filtered by bot."""
        pool = (
            self.session.query(GamePool)
            .filter_by(config_id=config_id, base_seed=base_seed)
            .first()
        )

        if not pool:
            return []

        return self.get_results_for_pool(pool.id, bot_id=bot_id, limit=num_games)


class DatabaseRepository:
    """Main repository that provides access to all sub-repositories."""

    def __init__(self, session: Session = None):
        self.session = session or db_manager.get_session()
        self.game_configs = GameConfigRepository(self.session)
        self.game_pools = GamePoolRepository(self.session)
        self.games = GameRepository(self.session)
        self.bots = BotRepository(self.session)
        self.results = GameResultRepository(self.session)

    def commit(self):
        """Commit the current transaction."""
        self.session.commit()

    def rollback(self):
        """Rollback the current transaction."""
        self.session.rollback()

    def close(self):
        """Close the session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        else:
            self.commit()
        self.close()