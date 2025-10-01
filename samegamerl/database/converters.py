"""Conversion utilities between pickle data and database models."""

import pickle
from pathlib import Path
from typing import Any

from samegamerl.database.repository import DatabaseRepository


def load_pickle_benchmark(filepath: Path | str) -> Any:
    """Load benchmark data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def convert_benchmark_data_to_db(benchmark_data: Any, repo: DatabaseRepository):
    """Convert BenchmarkData from pickle to database format.
    benchmark data has the format
    data_dict = {
            "games": data.games,
            "results": data.results,
            "config": data.config,
            "num_games": data.num_games,
            "base_seed": data.base_seed,
        }

    Args:
        benchmark_data: The loaded BenchmarkData object from pickle
        repo: Database repository for operations

    Returns:
        benchmark_set_id: ID of the created benchmark set
    """
    # Extract data from benchmark_data (dictionary format)
    games = benchmark_data['games']
    config = benchmark_data['config']
    num_games = benchmark_data['num_games']
    base_seed = benchmark_data['base_seed']

    # Create or find the game configuration
    game_config = repo.game_configs.find_or_create(
        num_rows=config.num_rows,
        num_cols=config.num_cols,
        num_colors=config.num_colors,
        name=f"{config.num_rows}x{config.num_cols}_{config.num_colors}colors"
    )

    # Create or find the game pool
    game_pool = repo.game_pools.find_or_create(
        config=game_config,
        base_seed=base_seed,
        max_games=num_games
    )

    # Flush to get the pool ID first
    repo.session.flush()

    # Create game records for each game in the pool (if not already created)
    existing_games = repo.games.get_games_for_pool(game_pool.id)
    games_to_create = len(games) - len(existing_games)

    if games_to_create > 0:
        start_index = len(existing_games)
        for i, game in enumerate(games[start_index:], start=start_index):
            repo.games.create_game(
                pool=game_pool,
                game_index=i,
                board_state=game.board if isinstance(game.board, list) else game.board.tolist(),
                seed=game.seed
            )

    # Flush to ensure we get the ID, then return it
    repo.session.flush()
    return game_pool.id


def convert_bot_results_to_db(bot_name: str, bot_type: str, results_data: list[dict],
                             pool_id: int, repo: DatabaseRepository) -> None:
    """Convert bot performance results to database format.

    Args:
        bot_name: Name of the bot
        bot_type: Type/category of the bot
        results_data: List of result dictionaries with performance metrics
        pool_id: ID of the game pool these results belong to
        repo: Database repository for operations
    """
    # Create or find the bot
    bot = repo.bots.find_or_create(bot_name, bot_type)

    # Get games for this pool (limited to the number of results we have)
    games = repo.games.get_games_for_pool(pool_id, limit=len(results_data))

    # Create results for each game
    for i, result in enumerate(results_data):
        if i < len(games):
            repo.results.create_result(
                game=games[i],
                bot=bot,
                tiles_cleared=result.get('tiles_cleared', 0),
                singles_remaining=result.get('singles_remaining', 0),
                moves_made=result.get('moves_made', 0),
                completed=result.get('completed', False)
            )


def export_benchmark_to_pickle(pool_id: int, num_games: int, output_path: Path | str,
                              repo: DatabaseRepository) -> None:
    """Export benchmark from database back to pickle format."""
    pool = repo.game_pools.get_by_id(pool_id)
    if not pool:
        raise ValueError(f"Game pool {pool_id} not found")

    games = repo.games.get_games_for_pool(pool_id, limit=num_games)

    # Reconstruct the original data structure
    # This would depend on your original BenchmarkData class structure
    benchmark_data = {
        'config': {
            'num_rows': pool.config.num_rows,
            'num_cols': pool.config.num_cols,
            'num_colors': pool.config.num_colors,
        },
        'num_games': len(games),
        'base_seed': pool.base_seed,
        'games': [
            {
                'game_index': game.game_index,
                'board_state': game.board_state,
                'seed': game.seed
            }
            for game in games
        ]
    }

    with open(output_path, 'wb') as f:
        pickle.dump(benchmark_data, f)


def migrate_pickle_directory(pickle_dir: Path | str, repo: DatabaseRepository) -> dict[str, int]:
    """Migrate all pickle files in a directory to database.

    Args:
        pickle_dir: Directory containing pickle files
        repo: Database repository for operations

    Returns:
        Dictionary mapping filename to pool_id
    """
    pickle_dir = Path(pickle_dir)
    migrated = {}

    for pickle_file in pickle_dir.glob("*.pkl"):
        try:
            benchmark_data = load_pickle_benchmark(pickle_file)
            pool_id = convert_benchmark_data_to_db(benchmark_data, repo)
            migrated[pickle_file.name] = pool_id
            print(f"✅ Migrated {pickle_file.name} -> pool_id {pool_id}")
        except Exception as e:
            print(f"❌ Failed to migrate {pickle_file.name}: {e}")

    return migrated