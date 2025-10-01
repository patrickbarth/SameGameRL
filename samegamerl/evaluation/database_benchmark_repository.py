"""Database-backed repository for benchmark data with memory efficiency."""

from samegamerl.database import DatabaseRepository, GamePool
from samegamerl.evaluation.benchmark_data import BenchmarkData, BotPerformance, GameSnapshot
from samegamerl.evaluation.benchmark_repository_interface import BenchmarkRepositoryInterface
from samegamerl.game.game_config import GameConfig


class DatabaseBenchmarkRepository(BenchmarkRepositoryInterface):
    """Database-backed repository for efficient benchmark data storage."""

    def __init__(self, config: GameConfig, base_seed: int):
        """Initialize with game configuration that defines this benchmark.

        Args:
            config: Game configuration (rows, cols, colors)
            base_seed: Base seed for game generation
        """
        self.config = config
        self.base_seed = base_seed
        self._pool_id = None
        self._game_config_id = None

    def save_data(self, data: BenchmarkData) -> None:
        """Save benchmark data to database."""
        with DatabaseRepository() as repo:
            # Create or find game configuration
            game_config = repo.game_configs.find_or_create(
                num_rows=data.config.num_rows,
                num_cols=data.config.num_cols,
                num_colors=data.config.num_colors,
                name=f"{data.config.num_rows}x{data.config.num_cols}_{data.config.num_colors}colors"
            )
            self._game_config_id = game_config.id

            # Create or extend game pool
            pool = repo.game_pools.find_or_create(
                config=game_config,
                base_seed=data.base_seed,
                max_games=data.num_games
            )
            repo.session.flush()
            self._pool_id = pool.id

            # Add games to pool (will skip if already exist)
            existing_games = repo.games.get_games_for_pool(pool.id)
            games_to_add = max(0, data.num_games - len(existing_games))

            if games_to_add > 0:
                start_index = len(existing_games)
                for i, game_snapshot in enumerate(data.games[start_index:], start=start_index):
                    repo.games.create_game(
                        pool=pool,
                        game_index=i,
                        board_state=game_snapshot.board,
                        seed=game_snapshot.seed
                    )

            # Save bot results using batch operations for performance
            for bot_name, performances in data.results.items():
                if not performances:
                    continue

                bot = repo.bots.find_or_create(bot_name, "benchmark_bot")

                # Get games for this pool
                games = repo.games.get_games_for_pool(pool.id, limit=len(performances))

                # Batch check: get all existing results for this bot/pool combination
                existing_results = repo.results.get_results_for_pool(pool.id, bot_id=bot.id)
                existing_game_indices = {result.game.game_index for result in existing_results}

                # Batch prepare: collect new results that don't exist yet
                new_results_to_create = []
                for i, performance in enumerate(performances):
                    if i < len(games) and i not in existing_game_indices:
                        new_results_to_create.append({
                            'game': games[i],
                            'bot': bot,
                            'tiles_cleared': performance.tiles_cleared,
                            'singles_remaining': performance.singles_remaining,
                            'moves_made': performance.moves_made,
                            'completed': performance.completed
                        })

                # Batch insert: create all new results at once
                for result_data in new_results_to_create:
                    repo.results.create_result(**result_data)

    def load_data(self) -> BenchmarkData | None:
        """Load benchmark data from database."""
        if not self.data_exists():
            return None

        with DatabaseRepository() as repo:
            # Get the game pool
            pool = repo.game_pools.get_by_config_and_seed(
                self._game_config_id or self._find_config_id(repo),
                self.base_seed
            )

            if not pool:
                return None

            self._pool_id = pool.id
            self._game_config_id = pool.config_id

            # Load games (limit to what's actually in the pool)
            db_games = repo.games.get_games_for_pool(pool.id)

            games = []
            for i, db_game in enumerate(db_games):
                games.append(GameSnapshot(
                    board=db_game.board_state,
                    config=self.config,
                    seed=db_game.seed,
                    game_id=i
                ))

            # Load all bot results for this pool
            results = {}

            # Get all bots that have results for this pool
            from samegamerl.database.models import Bot, GameResult, Game
            bots_with_results = (
                repo.session.query(Bot)
                .join(GameResult)
                .join(Game)
                .filter(Game.pool_id == pool.id)
                .distinct()
                .all()
            )

            for bot in bots_with_results:
                bot_results = repo.results.get_results_for_pool(pool.id, bot_id=bot.id)

                performances = []
                for result in bot_results:
                    performances.append(BotPerformance(
                        bot_name=bot.name,
                        game_id=result.game.game_index,
                        tiles_cleared=result.tiles_cleared,
                        singles_remaining=result.singles_remaining,
                        moves_made=result.moves_made,
                        completed=result.completed
                    ))

                if performances:
                    results[bot.name] = performances

            return BenchmarkData(
                games=games,
                results=results,
                config=self.config,
                num_games=len(games),
                base_seed=self.base_seed
            )

    def validate_results(self, bot_name: str, results: list[BotPerformance]) -> int:
        """Validate existing results for a bot and return count of valid consecutive results."""
        if not results:
            return 0

        valid_count = 0
        expected_game_id = 0

        for result in results:
            # Check game_id continuity (must be sequential starting from 0)
            if result.game_id != expected_game_id:
                break

            # Check bot_name consistency
            if result.bot_name != bot_name:
                break

            # Check that result has all required fields
            if not self._is_valid_performance(result):
                break

            valid_count += 1
            expected_game_id += 1

        return valid_count

    def determine_missing_games(
        self, bot_name: str, results: dict[str, list[BotPerformance]], num_games: int
    ) -> list[int]:
        """Determine which games need to be computed for a bot."""
        existing_results = results.get(bot_name, [])
        valid_count = self.validate_results(bot_name, existing_results)

        # Return list of missing game_ids
        return list(range(valid_count, num_games))

    def merge_results(
        self,
        existing: list[BotPerformance],
        new: list[BotPerformance],
        num_games: int,
        bot_name: str,
    ) -> list[BotPerformance]:
        """Merge new results with existing validated results for a bot."""
        valid_count = self.validate_results(bot_name, existing)

        # Keep only the valid existing results
        validated_existing = existing[:valid_count]

        # Create a dictionary for fast lookup of new results by game_id
        new_results_dict = {result.game_id: result for result in new}

        # Build final results list maintaining order
        merged_results = []

        for game_id in range(num_games):
            if game_id < valid_count:
                # Use existing valid result
                merged_results.append(validated_existing[game_id])
            elif game_id in new_results_dict:
                # Use new result
                merged_results.append(new_results_dict[game_id])
            else:
                # This shouldn't happen if determine_missing_games worked correctly
                # but we'll handle it gracefully
                break

        return merged_results

    def data_exists(self) -> bool:
        """Check if benchmark data exists in database."""
        with DatabaseRepository() as repo:
            config_id = self._game_config_id or self._find_config_id(repo)
            if not config_id:
                return False

            pool = repo.game_pools.get_by_config_and_seed(config_id, self.base_seed)
            return pool is not None

    def is_compatible(self, config: GameConfig, base_seed: int) -> bool:
        """Check if existing data is compatible with current configuration."""
        return (
            self.config.num_rows == config.num_rows
            and self.config.num_cols == config.num_cols
            and self.config.num_colors == config.num_colors
            and self.base_seed == base_seed
        )

    def _find_config_id(self, repo: DatabaseRepository) -> int | None:
        """Find the game config ID for this configuration."""
        config = repo.game_configs.find_or_create(
            num_rows=self.config.num_rows,
            num_cols=self.config.num_cols,
            num_colors=self.config.num_colors
        )
        repo.session.flush()
        return config.id

    def _is_valid_performance(self, result: BotPerformance) -> bool:
        """Check if a performance result has valid data."""
        try:
            return (
                isinstance(result.tiles_cleared, int)
                and result.tiles_cleared >= 0
                and isinstance(result.singles_remaining, int)
                and result.singles_remaining >= 0
                and isinstance(result.moves_made, int)
                and result.moves_made >= 0
                and isinstance(result.completed, bool)
            )
        except AttributeError:
            return False