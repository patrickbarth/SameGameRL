# Optional Database Dependencies

This project supports optional database storage for benchmarks, allowing it to work on remote GPU machines where database packages may not be available.

## Installation

### Core Dependencies (Required)
```bash
poetry install
```

This installs all core dependencies needed for training, evaluation, and visualization.

### With Database Support (Optional)
```bash
poetry install -E database
```

This additionally installs:
- `asyncpg` - Async PostgreSQL driver
- `sqlalchemy` - ORM and database abstraction
- `alembic` - Database migrations
- `psycopg2-binary` - PostgreSQL adapter
- `python-dotenv` - Environment variable management

## How It Works

The system automatically detects whether database dependencies are available:

- **With database packages**: You can use `storage_type='database'` in benchmarks
- **Without database packages**: System automatically falls back to `storage_type='pickle'`

### Check Availability

Run the test script to check current status:
```bash
poetry run python scripts/check_database_availability.py
```

### Example Output

**With database installed:**
```
============================================================
Database Dependency Status
============================================================
Database available: True
✓ All database dependencies are installed
  You can use storage_type='database'

Supported storage types: ['pickle', 'database']
============================================================
```

**Without database installed:**
```
============================================================
Database Dependency Status
============================================================
Database available: False
✗ Database dependencies not available
  Import error: No module named 'asyncpg'
  Install with: poetry install -E database

Supported storage types: ['pickle']
============================================================
```

## Usage in Code

The benchmark system handles this automatically:

```python
from samegamerl.evaluation.benchmark import Benchmark
from samegamerl.game.game_config import GameFactory

config = GameFactory.small()

# Request database storage
benchmark = Benchmark(
    config=config,
    num_games=100,
    storage_type='database'  # Will fallback to pickle if DB unavailable
)
```

If database dependencies are not installed, you'll see:
```
Warning: Database dependencies not available. Falling back to pickle storage.
Install database support with: poetry install -E database
```

## Use Cases

### Local Development (with database)
```bash
poetry install -E database
# Full database support for large-scale benchmarking
```

### Remote GPU Notebooks (without database)
```bash
poetry install
# Lighter installation, automatic fallback to pickle storage
```

## Implementation Details

The optional dependency pattern is implemented through:

1. **Availability Check** ([samegamerl/database/availability.py](samegamerl/database/availability.py))
   - Tries to import database packages at startup
   - Sets `DATABASE_AVAILABLE` flag

2. **Lazy Imports** ([samegamerl/database/__init__.py](samegamerl/database/__init__.py))
   - Only imports database modules if dependencies available
   - Prevents import errors when packages missing

3. **Factory Fallback** ([samegamerl/evaluation/benchmark_repository_factory.py](samegamerl/evaluation/benchmark_repository_factory.py))
   - Automatically switches to pickle storage when database unavailable
   - Provides clear user feedback

This pattern is common in data science libraries (pandas, scikit-learn, etc.) where features gracefully degrade based on available dependencies.