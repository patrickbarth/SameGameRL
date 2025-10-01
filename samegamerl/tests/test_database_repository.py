"""Unit tests for database repository operations."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from samegamerl.database.models import Base, Bot, GamePool, Game, GameConfig, GameResult
from samegamerl.database.repository import DatabaseRepository


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


@pytest.fixture
def repo(in_memory_db):
    """Create a repository with in-memory database."""
    return DatabaseRepository(session=in_memory_db)


class TestGameConfigRepository:
    def test_find_or_create_new_config(self, repo):
        """Test creating a new game configuration."""
        config = repo.game_configs.find_or_create(8, 8, 3, "Medium Game")

        assert config.num_rows == 8
        assert config.num_cols == 8
        assert config.num_colors == 3
        assert config.name == "Medium Game"
        assert config.id is not None

    def test_find_or_create_existing_config(self, repo):
        """Test finding an existing game configuration."""
        # Create first config
        config1 = repo.game_configs.find_or_create(8, 8, 3, "Medium Game")
        config1_id = config1.id

        # Try to create same config again
        config2 = repo.game_configs.find_or_create(8, 8, 3, "Different Name")

        # Should return the same config, not create a new one
        assert config2.id == config1_id
        assert config1 is config2

    def test_get_by_id(self, repo):
        """Test retrieving config by ID."""
        config = repo.game_configs.find_or_create(5, 5, 2)
        retrieved = repo.game_configs.get_by_id(config.id)

        assert retrieved is not None
        assert retrieved.id == config.id
        assert retrieved.num_rows == 5


class TestGamePoolRepository:
    def test_find_or_create_new_pool(self, repo):
        """Test creating a new game pool."""
        config = repo.game_configs.find_or_create(8, 8, 3)
        pool = repo.game_pools.find_or_create(config, 42, 100)

        assert pool.config_id == config.id
        assert pool.base_seed == 42
        assert pool.max_games == 100

    def test_find_or_create_existing_pool(self, repo):
        """Test finding an existing game pool."""
        config = repo.game_configs.find_or_create(8, 8, 3)
        pool1 = repo.game_pools.find_or_create(config, 42, 100)
        pool2 = repo.game_pools.find_or_create(config, 42, 100)

        assert pool1.id == pool2.id
        assert pool1 is pool2

    def test_extend_existing_pool(self, repo):
        """Test extending an existing pool with more games."""
        config = repo.game_configs.find_or_create(8, 8, 3)
        pool1 = repo.game_pools.find_or_create(config, 42, 50)
        pool2 = repo.game_pools.find_or_create(config, 42, 100)

        assert pool1.id == pool2.id
        assert pool2.max_games == 100  # Should be extended


class TestGameRepository:
    def test_create_game(self, repo):
        """Test creating a game."""
        config = repo.game_configs.find_or_create(5, 5, 2)
        pool = repo.game_pools.find_or_create(config, 42, 10)

        board_state = [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
        game = repo.games.create_game(pool, 0, board_state, 123)

        assert game.pool_id == pool.id
        assert game.game_index == 0
        assert game.board_state == board_state
        assert game.seed == 123

    def test_get_games_for_pool(self, repo):
        """Test retrieving all games for a pool."""
        config = repo.game_configs.find_or_create(5, 5, 2)
        pool = repo.game_pools.find_or_create(config, 42, 3)

        # Create multiple games
        game1 = repo.games.create_game(pool, 0, [[1, 2]], 100)
        game2 = repo.games.create_game(pool, 1, [[2, 1]], 101)
        game3 = repo.games.create_game(pool, 2, [[1, 1]], 102)

        games = repo.games.get_games_for_pool(pool.id)

        assert len(games) == 3
        assert games[0].game_index == 0
        assert games[1].game_index == 1
        assert games[2].game_index == 2

    def test_get_benchmark_games(self, repo):
        """Test retrieving subset of games for benchmark."""
        config = repo.game_configs.find_or_create(5, 5, 2)
        pool = repo.game_pools.find_or_create(config, 42, 5)

        # Create 5 games
        for i in range(5):
            repo.games.create_game(pool, i, [[i]], 100 + i)

        # Get first 3 games for benchmark
        benchmark_games = repo.games.get_benchmark_games(config.id, 42, 3)

        assert len(benchmark_games) == 3
        assert benchmark_games[0].game_index == 0
        assert benchmark_games[2].game_index == 2


class TestBotRepository:
    def test_find_or_create_new_bot(self, repo):
        """Test creating a new bot."""
        bot = repo.bots.find_or_create("RandomBot", "random")

        assert bot.name == "RandomBot"
        assert bot.bot_type == "random"
        assert bot.id is not None

    def test_find_or_create_existing_bot(self, repo):
        """Test finding an existing bot."""
        bot1 = repo.bots.find_or_create("RandomBot", "random")
        bot2 = repo.bots.find_or_create("RandomBot", "different_type")  # Same name

        assert bot1.id == bot2.id
        assert bot1 is bot2

    def test_get_by_name(self, repo):
        """Test retrieving bot by name."""
        bot = repo.bots.find_or_create("GreedyBot", "greedy")
        retrieved = repo.bots.get_by_name("GreedyBot")

        assert retrieved is not None
        assert retrieved.id == bot.id
        assert retrieved.name == "GreedyBot"


class TestGameResultRepository:
    def test_create_result(self, repo):
        """Test creating a game result."""
        # Setup dependencies
        config = repo.game_configs.find_or_create(5, 5, 2)
        pool = repo.game_pools.find_or_create(config, 42, 1)
        game = repo.games.create_game(pool, 0, [[1, 2]], 100)
        bot = repo.bots.find_or_create("TestBot", "test")

        # Create result
        result = repo.results.create_result(game, bot, 15, 3, 8, True)

        assert result.game_id == game.id
        assert result.bot_id == bot.id
        assert result.tiles_cleared == 15
        assert result.singles_remaining == 3
        assert result.moves_made == 8
        assert result.completed is True

    def test_get_results_for_bot(self, repo):
        """Test retrieving all results for a specific bot."""
        # Setup
        config = repo.game_configs.find_or_create(5, 5, 2)
        pool = repo.game_pools.find_or_create(config, 42, 2)
        game1 = repo.games.create_game(pool, 0, [[1, 2]], 100)
        game2 = repo.games.create_game(pool, 1, [[2, 1]], 101)
        bot = repo.bots.find_or_create("TestBot", "test")

        # Create results
        result1 = repo.results.create_result(game1, bot, 10, 2, 5, False)
        result2 = repo.results.create_result(game2, bot, 20, 0, 10, True)

        results = repo.results.get_results_for_bot(bot.id)

        assert len(results) == 2
        assert result1 in results
        assert result2 in results

    def test_get_results_for_pool(self, repo):
        """Test retrieving results for a game pool."""
        # Setup
        config = repo.game_configs.find_or_create(5, 5, 2)
        pool = repo.game_pools.find_or_create(config, 42, 1)
        game = repo.games.create_game(pool, 0, [[1, 2]], 100)
        bot1 = repo.bots.find_or_create("Bot1", "type1")
        bot2 = repo.bots.find_or_create("Bot2", "type2")

        # Create results
        result1 = repo.results.create_result(game, bot1, 10, 2, 5, False)
        result2 = repo.results.create_result(game, bot2, 15, 1, 7, True)

        # Test getting all results for pool
        all_results = repo.results.get_results_for_pool(pool.id)
        assert len(all_results) == 2

        # Test getting results for specific bot
        bot1_results = repo.results.get_results_for_pool(pool.id, bot_id=bot1.id)
        assert len(bot1_results) == 1
        assert bot1_results[0].bot_id == bot1.id

    def test_get_benchmark_results(self, repo):
        """Test retrieving results for a benchmark subset."""
        # Setup
        config = repo.game_configs.find_or_create(5, 5, 2)
        pool = repo.game_pools.find_or_create(config, 42, 3)

        # Create 3 games
        for i in range(3):
            repo.games.create_game(pool, i, [[i]], 100 + i)

        bot = repo.bots.find_or_create("TestBot", "test")

        # Create results for all 3 games
        games = repo.games.get_games_for_pool(pool.id)
        for game in games:
            repo.results.create_result(game, bot, 10 + game.game_index, 1, 5, True)

        # Test getting results for first 2 games only (benchmark)
        benchmark_results = repo.results.get_benchmark_results(config.id, 42, 2, bot.id)
        assert len(benchmark_results) == 2
        assert benchmark_results[0].tiles_cleared == 10  # First game
        assert benchmark_results[1].tiles_cleared == 11  # Second game


class TestDatabaseRepositoryContextManager:
    def test_context_manager_commit(self, in_memory_db):
        """Test repository context manager commits on success."""
        with DatabaseRepository(session=in_memory_db) as repo:
            config = repo.game_configs.find_or_create(5, 5, 2)
            config_id = config.id

        # Verify data was committed by checking in a new session
        with DatabaseRepository(session=in_memory_db) as repo:
            retrieved = repo.game_configs.get_by_id(config_id)
            assert retrieved is not None

    def test_context_manager_rollback(self, in_memory_db):
        """Test repository context manager rolls back on exception."""
        config_id = None

        try:
            with DatabaseRepository(session=in_memory_db) as repo:
                config = repo.game_configs.find_or_create(5, 5, 2)
                config_id = config.id
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Verify data was rolled back
        with DatabaseRepository(session=in_memory_db) as repo:
            retrieved = repo.game_configs.get_by_id(config_id) if config_id else None
            assert retrieved is None