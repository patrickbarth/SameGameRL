# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commit Message Guidelines

- **NEVER include the Claude Code attribution footer in commit messages**
- Focus on clear, descriptive commit messages that explain the changes
- Follow the existing project commit message style (see git log for examples)

## Test-Driven Development Workflow

**For all new features and bug fixes, follow this TDD process:**

1. **Development Phase** (multiple WIP commits):
   ```bash
   git commit -m "WIP: Add failing test for [feature/fix]"
   git commit -m "WIP: Implement [feature/fix]" 
   git commit -m "WIP: Refactor [improvement]"
   ```

2. **Before Sharing** (clean up with interactive rebase):
   ```bash
   git rebase -i HEAD~N  # Squash WIP commits into logical units
   git commit -m "Add [feature] with comprehensive tests"
   ```

**Key Principles:**
- Always write failing tests first (red)
- Implement minimal code to pass tests (green) 
- Refactor while keeping tests green
- Each final commit should pass all tests (bisectable history)
- Use WIP commits during development, clean commits for sharing

## Development Commands

### Testing
```bash
pytest samegamerl/tests/
pytest samegamerl/tests/test_samegame_env.py  # Run specific test file
```

### Dependencies
This project uses Poetry for dependency management:
```bash
poetry install              # Install core dependencies
poetry install -E database  # Install with optional database support
poetry shell               # Activate virtual environment
```

#### Optional Database Dependencies
The project supports optional database storage for benchmarks. Install database dependencies with:
```bash
poetry install -E database
```

Without database dependencies, the system automatically falls back to pickle-based storage. This allows the repository to work on remote GPU machines where database packages may not be available.

Check database availability:
```bash
poetry run python scripts/check_database_availability.py
```

### Running the Game
```bash
python samegamerl/main.py  # Launch interactive pygame interface
```

## Code Architecture

### Core Components

**Game Engine** (`samegamerl/game/`)
- `game.py`: Core SameGame logic, board manipulation, move validation
- `game_config.py`: Configuration system with GameConfig dataclass and GameFactory for different game sizes
- `game_params.py`: Display constants (colors, tile sizes, etc.)
- `Tile.py`, `View.py`: Pygame visualization components

**RL Environment** (`samegamerl/environments/`)
- `samegame_env.py`: OpenAI Gym-style environment wrapper
- Accepts GameConfig for flexible board dimensions and color counts
- Converts game board to one-hot encoded tensors (shape: [num_colors, num_rows, num_cols])
- Reward function prioritizes reducing singles count and clearing the board

**Agents** (`samegamerl/agents/`)
- `base_agent.py`: Abstract base class for all agents
- `dqn_agent.py`: Deep Q-Network implementation with configurable input/output dimensions
- `replay_buffer.py`: Experience replay buffer for DQN training

**Training Pipeline** (`samegamerl/training/`)
- `train.py`: Main training function with configurable epochs, reporting, and visualization
- Supports epsilon decay, target network updates, and periodic evaluation
- `checkpoint_data.py`: Typed dataclasses for checkpoint state (AgentCheckpointState, EnvCheckpointState, TrainingState, CheckpointData)
- `checkpoint_state_extractor.py`: Adapter pattern for extracting state from agent/env without coupling
- `checkpoint_service.py`: Coordinates checkpoint creation and loading
- `pickle_checkpoint_repository.py`: File-based checkpoint storage (zero dependencies)
- `training_manager.py`: High-level training interface with checkpoint support

**Experiments** (`samegamerl/experiments/`)
- Self-contained experiment scripts that define model architecture, hyperparameters, and training loops
- `conv_model.py`: CNN-based DQN experiment example
- `pyramid_model.py`: Fully connected DQN experiment example
- Each experiment includes model definition, training execution, and evaluation
- Models are now parameterized by GameConfig for flexible architectures

**Evaluation** (`samegamerl/evaluation/`)
- `validator.py`: Batch evaluation of trained agents
- `visualize_agent.py`: Interactive game visualization with agent play
- `plot_helper.py`: Training and evaluation plotting utilities

### Key Design Patterns

**Configuration-Driven Design**: Game dimensions and parameters are managed through GameConfig objects and factory methods, eliminating global constants and enabling flexible experimentation.

**Experiment-Driven Training**: Models and hyperparameters are defined within experiment files rather than in the DQN agent class. This allows for clean experiment comparison and reproducibility.

**Modular Architecture**: Clear separation between game logic, RL environment, agent implementations, and evaluation tools.

**Model Persistence**: Trained models are saved to `samegamerl/models/` with `.pth` extension, including model state, optimizer state, and target model state.

**Checkpoint System**: Comprehensive checkpoint tracking for resumable training:
- **Typed State Dataclasses**: Type-safe state representation (AgentCheckpointState, EnvCheckpointState, TrainingState, CheckpointData)
- **Adapter Pattern**: CheckpointStateExtractor decouples checkpoint system from domain classes
- **Repository Pattern**: Abstracted storage backend (currently pickle-based, extensible to database)
- **Service Decomposition**: Separate concerns (CheckpointService, TrainingOrchestrator, TrainingManager)
- **Version Field**: Checkpoint format evolution without breaking compatibility

## Code Style Guidelines

### Python Version & Features
- **Target Version**: Python 3.13+
- **Type Hints**: Use built-in types (`list`, `dict`, `tuple`) instead of `typing` module
- **Modern Features**: Leverage newer Python features when applicable

### Clean Code & Documentation
- **Self-Documenting Code**: Use clear, descriptive names that eliminate need for comments
- **Minimal Docstrings**: Only add docstrings that provide real value beyond what the code already expresses
- **Avoid Obvious Comments**: Don't document what the code clearly shows (e.g., `def total_cells()` doesn't need "Returns total cells")
- **Comment "Why", Not "What"**: When comments are needed, explain reasoning, not mechanics
- **Pythonic Solutions**: Prefer idiomatic Python patterns and built-in functions

### Game Configurations

The system supports multiple game configurations through GameFactory:
- **Small**: 5x5 board with 2 colors - `GameFactory.small()`
- **Medium**: 8x8 board with 3 colors - `GameFactory.medium()` (default)
- **Large**: 15x15 board with 5 colors - `GameFactory.large()`
- **Custom**: Any dimensions - `GameFactory.custom(rows, cols, colors)`

### Game Mechanics

SameGame is a tile-clearing puzzle where players click connected groups of same-colored tiles. The RL environment converts 2D coordinates to 1D actions via `divmod(action, config.num_cols)`. The reward function encourages reducing the number of single isolated tiles and completely clearing the board.

### Usage Examples

**Creating Different Game Sizes:**
```python
from samegamerl.game.game_config import GameFactory
from samegamerl.environments.samegame_env import SameGameEnv
from samegamerl.agents.dqn_agent import DqnAgent

# Small game for quick experiments
small_config = GameFactory.small()
env = SameGameEnv(small_config)

# Large game for complex scenarios
large_config = GameFactory.large()
env = SameGameEnv(large_config)

# Agent with appropriate dimensions
agent = DqnAgent(
    model=YourModel(config),
    config=config,
    # ... other parameters
)
```

**Training with Checkpoints:**
```python
from pathlib import Path
from samegamerl.training.checkpoint_service import CheckpointService
from samegamerl.training.pickle_checkpoint_repository import PickleCheckpointRepository
from samegamerl.training.training_manager import TrainingManager

# Set up checkpoint system (all files stored in one directory)
checkpoint_dir = Path("checkpoints")
repository = PickleCheckpointRepository(checkpoint_dir)
checkpoint_service = CheckpointService(repository)

# Create training manager
manager = TrainingManager(
    agent=agent,
    env=env,
    experiment_name="my_experiment",
    checkpoint_service=checkpoint_service
)

# Train with checkpoints every 1000 epochs
loss_history = manager.train_with_checkpoints(
    total_epochs=10000,
    checkpoint_every=1000,
    random_seed=42
)

# List all checkpoints for experiment
checkpoints = repository.list_checkpoints(experiment_name="my_experiment")
# ['my_experiment_epoch_1000', 'my_experiment_epoch_2000', ...]

# Load a specific checkpoint
checkpoint = checkpoint_service.load_checkpoint("my_experiment_epoch_5000")
print(f"Epsilon at epoch 5000: {checkpoint.agent_state.epsilon}")
print(f"Training progress: {checkpoint.training_state.current_epoch}/{checkpoint.training_state.total_epochs}")
```

**Training without Checkpoints:**
```python
from samegamerl.training.training_manager import TrainingManager

# Create training manager without checkpoint service
manager = TrainingManager(
    agent=agent,
    env=env,
    experiment_name="quick_test"
)

# Train without checkpointing
loss_history = manager.train(epochs=1000)
```
- Use "pythonic" ways of solving problems.
This project uses Python 3.13 or higher. Use features available in the newer versions of python when applicable
Don't use the typing module but use the built-in type declarations
- install and require packages through poetry by using the shell
- Use test-driven development and adhere to Clean Code principles
