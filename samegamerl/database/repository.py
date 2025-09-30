"""Repository pattern for database operations."""

from typing import Any, Optional

from sqlalchemy.orm import Session

from samegamerl.database.connection import db_manager
from samegamerl.database.models import Bot, BenchmarkSet, Game, GameConfig, GameResult


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


class BenchmarkSetRepository:
    def __init__(self, session: Session):
        self.session = session

    def find_or_create(self, config: GameConfig, num_games: int, base_seed: int) -> BenchmarkSet:
        """Find existing benchmark set or create new one."""
        benchmark_set = (
            self.session.query(BenchmarkSet)
            .filter_by(config_id=config.id, num_games=num_games, base_seed=base_seed)
            .first()
        )

        if not benchmark_set:
            benchmark_set = BenchmarkSet(
                config_id=config.id,
                num_games=num_games,
                base_seed=base_seed
            )
            self.session.add(benchmark_set)
            self.session.flush()

        return benchmark_set

    def get_by_id(self, set_id: int) -> Optional[BenchmarkSet]:
        """Get benchmark set by ID."""
        return self.session.query(BenchmarkSet).filter_by(id=set_id).first()


class GameRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_game(self, benchmark_set: BenchmarkSet, game_index: int,
                   board_state: list[list[int]], seed: int) -> Game:
        """Create a new game."""
        game = Game(
            benchmark_set_id=benchmark_set.id,
            game_index=game_index,
            board_state=board_state,  # SQLAlchemy will JSON-serialize this
            seed=seed
        )
        self.session.add(game)
        self.session.flush()
        return game

    def get_games_for_benchmark(self, benchmark_set_id: int) -> list[Game]:
        """Get all games for a benchmark set."""
        return (
            self.session.query(Game)
            .filter_by(benchmark_set_id=benchmark_set_id)
            .order_by(Game.game_index)
            .all()
        )


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

    def get_results_for_benchmark(self, benchmark_set_id: int, bot_id: int = None) -> list[GameResult]:
        """Get results for a benchmark set, optionally filtered by bot."""
        query = (
            self.session.query(GameResult)
            .join(Game)
            .filter(Game.benchmark_set_id == benchmark_set_id)
        )

        if bot_id:
            query = query.filter(GameResult.bot_id == bot_id)

        return query.all()


class DatabaseRepository:
    """Main repository that provides access to all sub-repositories."""

    def __init__(self, session: Session = None):
        self.session = session or db_manager.get_session()
        self.game_configs = GameConfigRepository(self.session)
        self.benchmark_sets = BenchmarkSetRepository(self.session)
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