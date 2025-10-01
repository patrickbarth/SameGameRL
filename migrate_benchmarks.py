#!/usr/bin/env python3
"""Migration script to demonstrate database efficiency over pickle files."""

import time
from pathlib import Path

from samegamerl.database import DatabaseRepository
from samegamerl.database.converters import migrate_pickle_directory, load_pickle_benchmark


def demonstrate_reuse_efficiency():
    """Demonstrate how the database eliminates duplicate game storage."""
    print("🔍 Demonstrating Database Efficiency vs Pickle Storage")
    print("=" * 60)

    # Check what files we have with the same config+seed but different sizes
    benchmark_dir = Path("samegamerl/evaluation/benchmarks")
    files = list(benchmark_dir.glob("*.pkl"))

    # Group by config and seed
    file_groups = {}
    for file in files:
        parts = file.stem.split("_")
        if len(parts) >= 5:  # benchmark_rows_cols_colors_seed
            key = f"{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}"  # rows_cols_colors_seed
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append(file)

    print(f"📊 Found {len(files)} total pickle files")
    print(f"📋 Grouped into {len(file_groups)} unique config+seed combinations:")

    total_pickle_size = 0
    for key, group in file_groups.items():
        group_size = sum(f.stat().st_size for f in group)
        total_pickle_size += group_size
        print(f"   {key}: {len(group)} files, {group_size / (1024*1024):.1f} MB")

    print(f"\n💾 Total pickle storage: {total_pickle_size / (1024*1024):.1f} MB")

    # Migrate one group to show efficiency
    largest_group = max(file_groups.values(), key=lambda g: len(g))
    print(f"\n🔄 Migrating largest group ({len(largest_group)} files):")

    with DatabaseRepository() as repo:
        migrated_pools = {}
        total_db_games = 0

        for file in largest_group:
            print(f"   📁 {file.name} ({file.stat().st_size / (1024*1024):.1f} MB)")

            start_time = time.time()
            data = load_pickle_benchmark(file)
            load_time = time.time() - start_time

            config = data['config']
            key = f"{config.num_rows}x{config.num_cols}_{config.num_colors}colors_seed{data['base_seed']}"

            if key not in migrated_pools:
                # First file for this config+seed - creates the pool
                start_migrate = time.time()
                pool_id = repo.game_pools.find_or_create(
                    config=repo.game_configs.find_or_create(
                        config.num_rows, config.num_cols, config.num_colors
                    ),
                    base_seed=data['base_seed'],
                    max_games=data['num_games']
                )
                repo.session.flush()

                # Add games to pool
                for i, game in enumerate(data['games']):
                    repo.games.create_game(
                        pool=pool_id,
                        game_index=i,
                        board_state=game.board if isinstance(game.board, list) else game.board.tolist(),
                        seed=game.seed
                    )

                migrate_time = time.time() - start_migrate
                migrated_pools[key] = pool_id
                total_db_games += data['num_games']
                print(f"      ✅ Created pool {pool_id.id} with {data['num_games']} games")
                print(f"      ⏱️  Load: {load_time:.2f}s, Migrate: {migrate_time:.2f}s")
            else:
                # Subsequent files - just extend pool if needed
                pool = migrated_pools[key]
                existing_games = len(repo.games.get_games_for_pool(pool.id))

                if data['num_games'] > existing_games:
                    # Extend pool with more games
                    pool.max_games = max(pool.max_games, data['num_games'])

                    for i in range(existing_games, data['num_games']):
                        repo.games.create_game(
                            pool=pool,
                            game_index=i,
                            board_state=data['games'][i].board if isinstance(data['games'][i].board, list) else data['games'][i].board.tolist(),
                            seed=data['games'][i].seed
                        )

                    added_games = data['num_games'] - existing_games
                    total_db_games += added_games
                    print(f"      📈 Extended pool {pool.id} with {added_games} more games")
                else:
                    print(f"      ♻️  Reused existing pool {pool.id} (no new games needed)")

                print(f"      ⏱️  Load: {load_time:.2f}s, Reuse: <0.01s")

        print(f"\n📊 Migration Results:")
        print(f"   🗂️  Created {len(migrated_pools)} game pools")
        print(f"   🎮 Total unique games stored: {total_db_games}")
        print(f"   💾 Pickle total size: {sum(f.stat().st_size for f in largest_group) / (1024*1024):.1f} MB")

        # Demonstrate querying different benchmark sizes
        print(f"\n🔍 Demonstrating Efficient Benchmark Queries:")
        if migrated_pools:
            pool_id = list(migrated_pools.values())[0]
            pool = repo.game_pools.get_by_id(pool_id.id)

            for size in [10, 50, 100, 500]:
                if size <= pool.max_games:
                    start_time = time.time()
                    games = repo.games.get_benchmark_games(
                        pool.config_id, pool.base_seed, size
                    )
                    query_time = time.time() - start_time
                    print(f"   📈 {size} games: {len(games)} retrieved in {query_time:.3f}s")


def migrate_all_benchmarks():
    """Migrate all benchmark pickle files to database."""
    print("\n🚀 Starting Full Migration of All Benchmark Files")
    print("=" * 60)

    benchmark_dir = Path("samegamerl/evaluation/benchmarks")

    with DatabaseRepository() as repo:
        migrated = {}

        # Get all pickle files
        pickle_files = list(benchmark_dir.glob("*.pkl"))
        print(f"📁 Found {len(pickle_files)} pickle files to migrate")

        start_time = time.time()

        for i, file in enumerate(pickle_files, 1):
            try:
                print(f"\n[{i:2d}/{len(pickle_files)}] 📄 {file.name}")

                file_start = time.time()
                data = load_pickle_benchmark(file)
                config = data['config']

                # Find or create pool (reuses if same config+seed exists)
                pool = repo.game_pools.find_or_create(
                    config=repo.game_configs.find_or_create(
                        config.num_rows, config.num_cols, config.num_colors,
                        name=f"{config.num_rows}x{config.num_cols}_{config.num_colors}colors"
                    ),
                    base_seed=data['base_seed'],
                    max_games=data['num_games']
                )
                repo.session.flush()

                # Add games (will skip if already exist)
                existing_games = len(repo.games.get_games_for_pool(pool.id))
                games_to_add = max(0, data['num_games'] - existing_games)

                if games_to_add > 0:
                    for j in range(existing_games, data['num_games']):
                        repo.games.create_game(
                            pool=pool,
                            game_index=j,
                            board_state=data['games'][j].board if isinstance(data['games'][j].board, list) else data['games'][j].board.tolist(),
                            seed=data['games'][j].seed
                        )

                file_time = time.time() - file_start
                file_size = file.stat().st_size / (1024 * 1024)

                migrated[file.name] = pool.id
                print(f"         ✅ Pool {pool.id}, {games_to_add} new games, {file_time:.2f}s, {file_size:.1f} MB")

            except Exception as e:
                print(f"         ❌ Failed: {e}")

        total_time = time.time() - start_time

        print(f"\n📊 Migration Complete!")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   ✅ Successfully migrated: {len(migrated)}/{len(pickle_files)} files")
        print(f"   🗂️  Unique pools created: {len(set(migrated.values()))}")


if __name__ == "__main__":
    print("🗄️  SameGameRL Benchmark Migration")
    print("=" * 60)

    # First demonstrate the efficiency gains
    demonstrate_reuse_efficiency()

    # Then optionally migrate everything
    print(f"\n" + "=" * 60)
    choice = input("\n🤔 Migrate all benchmark files to database? (y/N): ").lower().strip()

    if choice in ['y', 'yes']:
        migrate_all_benchmarks()
        print("\n✨ Migration complete! Your benchmark system now uses the database.")
    else:
        print("\n👍 Demo complete! Run again with 'y' to perform full migration.")