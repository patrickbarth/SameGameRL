# SameGameRL Architecture Documentation

## System Overview

SameGameRL implements a complete reinforcement learning solution for the SameGame puzzle using Deep Q-Networks (DQN). The architecture prioritizes modularity, configurability, and extensibility while maintaining clean separation of concerns.

## Core Design Principles

### 1. Configuration-Driven Architecture

**Design Decision**: All game parameters (board size, colors) flow through a centralized `GameConfig` system rather than global constants.

```python
@dataclass
class GameConfig:
    num_rows: int = 8
    num_cols: int = 8  
    num_colors: int = 4
    
    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.num_colors, self.num_rows, self.num_cols)
```

**Rationale**: 
- Enables systematic experimentation across game variants
- Eliminates global state and magic numbers
- Simplifies model architecture parameterization
- Supports clean unit testing with controlled configurations

**Impact**: Models automatically adjust input/output dimensions based on configuration, enabling seamless scaling from 5x5 to 15x15 boards.

### 2. OpenAI Gym-Compatible Environment

**Design Decision**: Implement standard `reset()`, `step()`, `get_observation()` interface despite not inheriting from gym.Env.

```python
class SameGameEnv:
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        # Convert 1D action to 2D coordinates
        row, col = self._to_2d(action)
        self.game.move((row, col))
        
        reward = self.compute_reward(...)
        return self.get_observation(), reward, self.done, {}
```

**Rationale**:
- Familiar interface for RL practitioners
- Easy integration with existing RL libraries
- Clean separation between game logic and RL environment
- Standardized API for different algorithms

### 3. One-Hot State Representation

**Design Decision**: Represent board state as one-hot encoded tensor with shape `[num_colors, num_rows, num_cols]`.

```python
def get_observation(self) -> np.ndarray:
    board_one_hot = np.zeros((self.num_colors, self.num_rows, self.num_cols))
    for row in range(self.num_rows):
        for col in range(self.num_cols):
            color = self.game.board[row][col]
            if color > 0:  # 0 represents empty cell
                board_one_hot[color - 1, row, col] = 1
    return board_one_hot
```

**Rationale**:
- **CNN Compatibility**: Natural channels-first format for convolutional networks
- **Sparse Representation**: Efficient encoding of discrete color information
- **Scalable**: Works seamlessly across different board sizes and color counts
- **No Embedding Required**: Direct network input without additional preprocessing

**Alternative Considered**: Flattened integer board representation was rejected due to poor CNN performance and unclear color relationships.

### 4. Reward Function

**Design Decision**: Multi-component reward function balancing progress and completion.

```python
def compute_reward(self, prev_left, cur_left, prev_singles, cur_singles, action) -> float:
    # Component 1: Tiles removed
    tiles_removed = prev_left - cur_left
    
    # Component 2: Singles reduction (key insight)
    singles_improvement = prev_singles - cur_singles
    
    # Component 3: Completion bonus
    completion_bonus = 1000.0 if cur_left == 0 else 0.0
    
    return (tiles_removed * 10.0) + (singles_improvement * 50.0) + completion_bonus
```

**Key Insight**: Prioritizing singles reduction over raw tile removal leads to better long-term strategy.

**Rationale**:
- **Strategic Depth**: Encourages planning beyond greedy tile removal  
- **Completion Incentive**: Large bonus for fully clearing the board
- **Balance**: Prevents local optima while rewarding progress
- **Empirically Validated**: Outperformed simpler reward functions in experiments

## Component Architecture

### Game Engine (`samegamerl/game/`)

**Responsibility**: Core SameGame mechanics independent of RL concerns.

```
game.py           # Board manipulation, move validation, game rules
game_config.py    # Configuration system with factory pattern
game_params.py    # Display constants (colors, sizes)
Tile.py, View.py  # Pygame visualization components
```

**Design Patterns**:
- **Factory Pattern**: `GameFactory` for standard configurations
- **Immutable Configuration**: GameConfig validates parameters once
- **Single Responsibility**: Pure game logic without RL coupling

### RL Environment (`samegamerl/environments/`)

**Responsibility**: Bridge between game engine and RL algorithms.

**Key Features**:
- **Action Space Conversion**: 1D actions → 2D board coordinates via `divmod()`
- **Reward Calculation**: Complex multi-component scoring system
- **State Normalization**: Consistent observation format across configurations
- **Episode Management**: Proper reset/termination handling

### Agent System (`samegamerl/agents/`)

**Design Decision**: Modular agent architecture with pluggable models.

```python
class DqnAgent(BaseAgent):
    def __init__(self, model: nn.Module, config: GameConfig, ...):
        self.model = model  # Pluggable architecture
        self.target_model = deepcopy(model)
        self.replay_buffer = ReplayBuffer(...)
```

**Benefits**:
- **Model Flexibility**: CNN, fully-connected, or custom architectures
- **Hyperparameter Isolation**: Configuration separated from model definition
- **Standard Interface**: `BaseAgent` ensures consistent API
- **Experience Replay**: Sophisticated buffer management with uniform sampling

### Experiment System (`samegamerl/experiments/`)

**Philosophy**: Models and hyperparameters defined within experiment files rather than agent classes.

```python
# experiments/conv_model.py
class ConvModel(nn.Module):
    def __init__(self, config: GameConfig):
        # Model architecture parameterized by config
        
def create_agent(config: GameConfig) -> DqnAgent:
    model = ConvModel(config)
    return DqnAgent(model, config, lr=1e-4, epsilon=0.1)
```

**Advantages**:
- **Reproducibility**: Complete experiment specification in single file
- **Comparison**: Easy to compare different approaches
- **Clean Separation**: Agent implementation independent of model choices
- **Configuration Driven**: Models automatically scale with game parameters

## Algorithm Selection: Why DQN?

**Decision Rationale**:

1. **Discrete Action Space**: SameGame has finite, discrete actions (board positions)
2. **Deterministic Environment**: No stochasticity in game mechanics
3. **Complex State Space**: Large state spaces (up to 6^225 for 15x15 board) benefit from function approximation
4. **Proven Architecture**: DQN well-established for discrete control problems
5. **Implementation Simplicity**: Straightforward to implement and debug

**Alternatives Considered**:
- **Policy Gradient Methods**: Rejected due to unnecessary complexity for discrete actions
- **Actor-Critic**: Overkill for deterministic environment  
- **Monte Carlo Methods**: Impractical for large state spaces
- **Tabular Q-Learning**: Impossible due to state space size

## Training Pipeline Architecture

**Design Decision**: Separate training logic from agent implementation.

```python
# training/train.py
def train(model, env, epochs=1000, batch_size=32, ...):
    agent = create_agent_from_model(model, env.config)
    
    for epoch in range(epochs):
        # Training loop with configurable parameters
        # Periodic evaluation and model saving
        # Epsilon decay scheduling
```

**Benefits**:
- **Flexibility**: Same training pipeline works with different agents/models
- **Monitoring**: Built-in evaluation and progress tracking
- **Reproducibility**: Consistent training across experiments
- **Hyperparameter Isolation**: Training parameters separated from model definition

## Testing Strategy

**Architecture**: Comprehensive test coverage across all components.

```
tests/
├── test_game.py           # Game logic validation
├── test_samegame_env.py   # Environment interface compliance
├── test_dqn_agent.py      # Agent behavior verification  
├── test_replay_buffer.py  # Experience replay correctness
└── test_train.py          # Training pipeline integration
```

**Testing Principles**:
- **Component Isolation**: Each component tested independently
- **Configuration Variation**: Tests across multiple game configurations
- **Integration Testing**: End-to-end training pipeline validation
- **Property-Based Testing**: State invariants and transitions verified

## Scalability Considerations

### Memory Management
- **Replay Buffer**: Configurable capacity with circular buffer implementation
- **Model Architecture**: Dynamic sizing based on game configuration
- **Batch Processing**: Efficient tensor operations for training

### Computational Efficiency
- **Action Masking**: Invalid actions filtered during selection
- **Target Network Updates**: Periodic rather than continuous updates
- **Vectorized Operations**: NumPy/PyTorch optimizations throughout

### Configuration Scalability
- **Dynamic Model Sizing**: Networks automatically adjust to game parameters
- **Memory Scaling**: Buffer and batch sizes adapt to problem complexity
- **Training Duration**: Automatic epoch scaling based on state space size

## Future Architecture Extensions

**Planned Enhancements**:
1. **Multi-Agent Support**: Competitive and cooperative scenarios
2. **Curriculum Learning**: Progressive difficulty scaling
3. **Distributed Training**: Multi-GPU and multi-node support
4. **Advanced Algorithms**: Double DQN, Dueling DQN, Rainbow DQN
5. **Real-time Inference**: Optimized models for interactive play

**Architecture Flexibility**: Current modular design supports these extensions without breaking existing interfaces.

---

This architecture represents a production-ready reinforcement learning system with emphasis on maintainability, extensibility, and empirical validation. The configuration-driven approach enables systematic experimentation while maintaining clean code organization.
