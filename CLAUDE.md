# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
pytest samegamerl/tests/
pytest samegamerl/tests/test_samegame_env.py  # Run specific test file
```

### Dependencies
This project uses Poetry for dependency management:
```bash
poetry add    # Install dependencies
poetry shell      # Activate virtual environment
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
- **Large**: 15x15 board with 6 colors - `GameFactory.large()`
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
    input_shape=config.observation_shape,
    action_space_size=config.action_space_size,
    # ... other parameters
)
```
- Use "pythonic" ways of solving problems.
This project uses Python 3.13 or higher. Use features available in the newer versions of python when applicable
Don't use the typing module but use the built-in type declarations
- install and require packages through poetry by using the shell