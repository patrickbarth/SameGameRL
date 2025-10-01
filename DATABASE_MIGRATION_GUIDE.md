# SameGameRL Database Migration Guide

## Executive Summary

**Problem**: Current pickle-based benchmark storage causes memory issues (2.9MB+ files loaded entirely) and poor performance as data grows.

**Solution**: Migrate to PostgreSQL database with hybrid SQL/JSON architecture for structured data + flexible ML metadata.

**Impact**: Enables selective queries, memory efficiency, concurrent access, and foundation for ML model versioning.

---

## Architectural Decisions

### Database Technology: PostgreSQL

**Why PostgreSQL over alternatives:**

| Aspect | PostgreSQL | SQLite | MongoDB | ClickHouse |
|--------|------------|---------|---------|------------|
| Concurrent access | ✅ Excellent | ❌ Poor | ✅ Good | ✅ Excellent |
| Complex queries | ✅ Excellent | ✅ Good | ❌ Limited | ✅ Excellent |
| Binary storage | ✅ BYTEA | ✅ BLOB | ✅ GridFS | ❌ Limited |
| JSON support | ✅ Native | ✅ JSON1 | ✅ Native | ❌ Limited |
| Python ecosystem | ✅ Mature | ✅ Built-in | ✅ Good | ❌ Limited |
| Analytics queries | ✅ Excellent | ❌ Poor | ❌ Poor | ✅ Excellent |

**Decision**: PostgreSQL provides the best balance of features for our ML/gaming workload.

### Schema Design Philosophy

**Hybrid Approach**:
- **Structured columns** for frequently queried data (performance metrics, timestamps)
- **JSON columns** for flexible metadata (hyperparameters, configurations)
- **Binary columns** for model weights and large data

**Benefits**:
- Query performance on structured data
- Schema flexibility for evolving ML experiments
- No complex object-relational mapping

---

## Database Schema

### Core Tables

```sql
-- Game configurations
CREATE TABLE game_configs (
    id SERIAL PRIMARY KEY,
    num_rows INTEGER NOT NULL,
    num_cols INTEGER NOT NULL,
    num_colors INTEGER NOT NULL,
    name VARCHAR(100) UNIQUE,  -- e.g., "small", "medium", "large"
    metadata JSON DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_dimensions CHECK (
        num_rows > 0 AND num_cols > 0 AND num_colors > 0
    )
);

-- Benchmark sets (groups of games for comparison)
CREATE TABLE benchmark_sets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    config_id INTEGER REFERENCES game_configs(id),
    num_games INTEGER NOT NULL,
    base_seed INTEGER NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(config_id, base_seed, num_games)
);

-- Individual games
CREATE TABLE games (
    id SERIAL PRIMARY KEY,
    benchmark_set_id INTEGER REFERENCES benchmark_sets(id),
    game_index INTEGER NOT NULL,  -- 0 to num_games-1
    board_state JSON NOT NULL,    -- Initial board as 2D array
    seed INTEGER NOT NULL,
    metadata JSON DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(benchmark_set_id, game_index)
);

-- Bot definitions
CREATE TABLE bots (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    bot_type VARCHAR(100) NOT NULL,  -- 'rule_based', 'dqn', 'human'
    version VARCHAR(50) DEFAULT '1.0',
    metadata JSON DEFAULT '{}',      -- Bot-specific parameters
    created_at TIMESTAMP DEFAULT NOW()
);

-- Game execution results
CREATE TABLE game_results (
    id SERIAL PRIMARY KEY,
    game_id INTEGER REFERENCES games(id) ON DELETE CASCADE,
    bot_id INTEGER REFERENCES bots(id),
    tiles_cleared INTEGER NOT NULL,
    singles_remaining INTEGER NOT NULL,
    moves_made INTEGER NOT NULL,
    completed BOOLEAN NOT NULL,
    execution_time_ms INTEGER,
    metadata JSON DEFAULT '{}',      -- Additional metrics
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(game_id, bot_id),
    CONSTRAINT valid_metrics CHECK (
        tiles_cleared >= 0 AND
        singles_remaining >= 0 AND
        moves_made >= 0 AND
        execution_time_ms >= 0
    )
);
```

### Future Extensions

```sql
-- Training sessions for ML agents
CREATE TABLE training_sessions (
    id SERIAL PRIMARY KEY,
    bot_id INTEGER REFERENCES bots(id),
    name VARCHAR(255),
    hyperparameters JSON NOT NULL,
    config_id INTEGER REFERENCES game_configs(id),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status VARCHAR(50) DEFAULT 'running',
    final_metrics JSON DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Model checkpoints and weights
CREATE TABLE model_checkpoints (
    id SERIAL PRIMARY KEY,
    training_session_id INTEGER REFERENCES training_sessions(id),
    epoch INTEGER NOT NULL,
    model_weights BYTEA,              -- Binary model data
    optimizer_state BYTEA,            -- Binary optimizer state
    metrics JSON DEFAULT '{}',        -- epoch metrics (loss, accuracy)
    file_path TEXT,                   -- Optional: path to .pth file
    file_size_bytes INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(training_session_id, epoch)
);

-- Detailed game move histories
CREATE TABLE game_histories (
    id SERIAL PRIMARY KEY,
    game_id INTEGER REFERENCES games(id) ON DELETE CASCADE,
    bot_id INTEGER REFERENCES bots(id),
    move_sequence JSON NOT NULL,      -- Array of (row, col) moves
    board_states JSON,                -- Optional: intermediate board states
    decision_metadata JSON DEFAULT '{}', -- Agent decision info
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(game_id, bot_id)
);
```

### Indexes for Performance

```sql
-- Query optimization indexes
CREATE INDEX idx_games_benchmark_set ON games(benchmark_set_id);
CREATE INDEX idx_games_seed ON games(seed);

CREATE INDEX idx_game_results_game_bot ON game_results(game_id, bot_id);
CREATE INDEX idx_game_results_bot ON game_results(bot_id);
CREATE INDEX idx_game_results_performance ON game_results(tiles_cleared, completed);

CREATE INDEX idx_benchmark_sets_config ON benchmark_sets(config_id);
CREATE INDEX idx_training_sessions_bot ON training_sessions(bot_id);
CREATE INDEX idx_model_checkpoints_session ON model_checkpoints(training_session_id);

-- JSON indexes for metadata queries
CREATE INDEX idx_bots_metadata_gin ON bots USING GIN (metadata);
CREATE INDEX idx_game_results_metadata_gin ON game_results USING GIN (metadata);
```

---

## Implementation Phases

### Phase 1: Database Setup and Core Schema
**Goal**: Establish database foundation

**Tasks**:
- [ ] Install PostgreSQL locally and create development database
- [ ] Add database dependencies to poetry.toml
- [ ] Create database connection management
- [ ] Implement core table schema
- [ ] Create basic database models with SQLAlchemy
- [ ] Write database initialization script

**Dependencies**: poetry, asyncpg, SQLAlchemy

**Estimated effort**: 1-2 days

### Phase 2: Data Models and Repository Layer
**Goal**: Abstract database operations

**Tasks**:
- [ ] Create SQLAlchemy ORM models for all tables
- [ ] Implement database repository pattern
- [ ] Create data conversion utilities (pickle ↔ database)
- [ ] Write unit tests for repository operations
- [ ] Implement connection pooling
- [ ] Add database configuration management

**Dependencies**: Phase 1 complete

**Estimated effort**: 2-3 days

### Phase 3: Migration from Pickle Files
**Goal**: Convert existing data to database

**Tasks**:
- [ ] Write migration script for existing benchmark data
- [ ] Create data validation and integrity checks
- [ ] Implement backup/rollback procedures
- [ ] Migrate existing .pkl files to database
- [ ] Verify data integrity after migration
- [ ] Performance comparison (old vs new system)

**Dependencies**: Phase 2 complete

**Estimated effort**: 1-2 days

### Phase 4: Update Benchmark System
**Goal**: Replace pickle-based storage in benchmark system

**Tasks**:
- [ ] Update BenchmarkRepository to use database
- [ ] Modify Benchmark class for database operations
- [ ] Update lazy loading to use database queries
- [ ] Add selective data loading (partial results)
- [ ] Update benchmark execution strategies
- [ ] Comprehensive testing of updated system

**Dependencies**: Phase 3 complete

**Estimated effort**: 2-3 days

### Phase 5: Performance Optimization
**Goal**: Optimize for production workloads

**Tasks**:
- [ ] Query optimization and index tuning
- [ ] Implement database query caching
- [ ] Add batch operations for large datasets
- [ ] Optimize binary data storage/retrieval
- [ ] Add database monitoring and metrics
- [ ] Performance benchmarking and tuning

**Dependencies**: Phase 4 complete

**Estimated effort**: 1-2 days

### Phase 6: Advanced Features (Future)
**Goal**: Enable ML model versioning and training history

**Tasks**:
- [ ] Implement training session tracking
- [ ] Add model checkpoint storage
- [ ] Create game history detailed tracking
- [ ] Build analytics dashboard queries
- [ ] Add data export/import capabilities
- [ ] Integration with training pipeline

**Dependencies**: Phase 5 complete

**Estimated effort**: 3-4 days

---

## Technical Specifications

### Dependencies

```toml
# Add to pyproject.toml [tool.poetry.dependencies]
asyncpg = "^0.29.0"          # PostgreSQL async driver
sqlalchemy = "^2.0.23"       # ORM framework
alembic = "^1.12.1"          # Database migrations
psycopg2-binary = "^2.9.7"   # PostgreSQL adapter
```

### Database Configuration

```python
# samegamerl/database/config.py
DATABASE_CONFIG = {
    "development": {
        "host": "localhost",
        "port": 5432,
        "database": "samegamerl_dev",
        "username": "postgres",
        "password": "postgres",
    },
    "test": {
        "host": "localhost",
        "port": 5432,
        "database": "samegamerl_test",
        "username": "postgres",
        "password": "postgres",
    }
}
```

### Performance Targets

**Current (Pickle) vs Target (Database)**:

| Operation | Current | Target | Improvement |
|-----------|---------|--------|-------------|
| Load benchmark data | 500ms-2s | <100ms | 5-20x |
| Query bot performance | N/A (full load) | <50ms | New capability |
| Memory usage | 2.9MB+ per benchmark | <10MB total | 100x+ |
| Concurrent access | File locks | Multi-user | New capability |
| Partial data loading | Impossible | <10ms | New capability |

---

## Migration Strategy

### Data Conversion Mapping

**BenchmarkData → Database Tables**:
```python
# Current structure
BenchmarkData:
  - games: list[GameSnapshot]     → games table
  - results: dict[str, list[BotPerformance]]  → game_results table
  - config: GameConfig            → game_configs table
  - num_games: int               → benchmark_sets.num_games
  - base_seed: int               → benchmark_sets.base_seed

# New structure enables:
# - Selective loading by bot, game, or configuration
# - Cross-benchmark queries and analytics
# - Incremental result updates
```

### Backward Compatibility

During migration:
1. **Dual storage**: Write to both pickle and database
2. **Gradual migration**: Convert one benchmark set at a time
3. **Fallback mechanism**: Automatic fallback to pickle if database fails
4. **Validation**: Compare results between old and new systems

### Testing Strategy

**Unit Tests**:
- Database model validation
- Repository CRUD operations
- Data conversion utilities
- Connection handling

**Integration Tests**:
- End-to-end benchmark execution
- Migration script validation
- Performance comparisons
- Concurrent access scenarios

**Load Tests**:
- Large dataset handling
- Memory usage under load
- Query performance with realistic data sizes

---

## Open Questions and Decisions Needed

### Configuration Management
- **Question**: Environment-specific database configuration approach?
- **Options**: Environment variables, config files, or command line args?
- **Recommendation**: Environment variables with config file fallback

### Binary Data Storage
- **Question**: Store model weights in database or filesystem?
- **Trade-offs**: Database = consistency, filesystem = performance
- **Recommendation**: Database for small models (<10MB), filesystem with path references for larger

### Migration Timing
- **Question**: Migrate all existing data or start fresh?
- **Impact**: All existing benchmarks vs clean slate
- **Recommendation**: Migrate critical benchmarks, archive old ones

### Deployment Environment
- **Question**: Local PostgreSQL vs cloud database?
- **Options**: Local development, cloud for production
- **Recommendation**: Local for development, plan for cloud scaling

---

## Success Metrics

### Performance Improvements
- [ ] Memory usage reduced by >90%
- [ ] Query response time <100ms for typical operations
- [ ] Support for concurrent benchmark execution
- [ ] Selective data loading functional

### Feature Enablement
- [ ] Cross-benchmark performance analysis
- [ ] Incremental result updates
- [ ] Model versioning infrastructure ready
- [ ] Game history tracking available

### System Reliability
- [ ] Zero data loss during migration
- [ ] Automatic backup and recovery
- [ ] Error handling and graceful degradation
- [ ] Comprehensive test coverage >90%

---

## Current Status

**Phase**: Planning Complete ✅
**Next Action**: Begin Phase 1 - Database Setup
**Estimated Total Effort**: 10-15 days
**Critical Path**: Schema design → Migration script → Benchmark system integration

---

*This document will be updated as implementation progresses. All architectural decisions and progress will be tracked here.*