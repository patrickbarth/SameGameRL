-- SameGameRL Database Schema

-- Game configurations (board dimensions and settings)
CREATE TABLE IF NOT EXISTS game_configs (
    id SERIAL PRIMARY KEY,
    num_rows INTEGER NOT NULL,
    num_cols INTEGER NOT NULL,
    num_colors INTEGER NOT NULL,
    name VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_dimensions CHECK (
        num_rows > 0 AND num_cols > 0 AND num_colors > 0
    ),
    UNIQUE(num_rows, num_cols, num_colors)
);

-- Benchmark sets (groups of games for comparison)
CREATE TABLE IF NOT EXISTS benchmark_sets (
    id SERIAL PRIMARY KEY,
    config_id INTEGER REFERENCES game_configs(id),
    num_games INTEGER NOT NULL,
    base_seed INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(config_id, base_seed, num_games)
);

-- Individual games within benchmark sets
CREATE TABLE IF NOT EXISTS games (
    id SERIAL PRIMARY KEY,
    benchmark_set_id INTEGER REFERENCES benchmark_sets(id),
    game_index INTEGER NOT NULL,
    board_state JSON NOT NULL,
    seed INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(benchmark_set_id, game_index)
);

-- Bot definitions
CREATE TABLE IF NOT EXISTS bots (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    bot_type VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Game execution results
CREATE TABLE IF NOT EXISTS game_results (
    id SERIAL PRIMARY KEY,
    game_id INTEGER REFERENCES games(id) ON DELETE CASCADE,
    bot_id INTEGER REFERENCES bots(id),
    tiles_cleared INTEGER NOT NULL,
    singles_remaining INTEGER NOT NULL,
    moves_made INTEGER NOT NULL,
    completed BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(game_id, bot_id),
    CONSTRAINT valid_metrics CHECK (
        tiles_cleared >= 0 AND
        singles_remaining >= 0 AND
        moves_made >= 0
    )
);

-- Basic indexes for common queries
CREATE INDEX IF NOT EXISTS idx_games_benchmark_set ON games(benchmark_set_id);
CREATE INDEX IF NOT EXISTS idx_game_results_game_bot ON game_results(game_id, bot_id);
CREATE INDEX IF NOT EXISTS idx_game_results_bot ON game_results(bot_id);