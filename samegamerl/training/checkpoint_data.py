"""Data structures for checkpoint tracking system.

This module defines typed dataclasses for storing checkpoint state:
- AgentCheckpointState: DQN agent hyperparameters and training state
- EnvCheckpointState: Environment configuration and reward function parameters
- TrainingState: Training loop metadata and progress tracking
- CheckpointData: Complete checkpoint snapshot combining all state

These structures enable type-safe serialization and clear contracts between
components without coupling domain classes to the checkpoint system.
"""

from dataclasses import dataclass, field
from datetime import datetime

from samegamerl.game.game_config import GameConfig


@dataclass
class AgentCheckpointState:
    """Snapshot of DQN agent hyperparameters and training state.

    Captures all parameters that may evolve during training, particularly
    epsilon which decays over time. Essential for resuming training from
    checkpoints without resetting exploration strategy.
    """

    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    learning_rate: float
    gamma: float
    tau: float
    batch_size: int
    replay_buffer_size: int


@dataclass
class EnvCheckpointState:
    """Snapshot of environment configuration and reward parameters.

    Stores parameterized reward function settings that may be tuned during
    training experiments. GameConfig ensures reproducibility of board
    dimensions and color counts.
    """

    completion_reward: float
    partial_completion_base: float
    invalid_move_penalty: float
    singles_reduction_weight: float
    game_config: GameConfig


@dataclass
class TrainingState:
    """Training loop metadata and progress tracking.

    Captures training progress and metadata necessary for resuming training
    or analyzing checkpoint context. Optional fields support extended metrics
    without breaking backward compatibility.
    """

    total_epochs: int
    current_epoch: int
    total_steps: int
    random_seed: int
    best_score: float | None = None
    training_time_seconds: float | None = None


@dataclass
class CheckpointData:
    """Complete checkpoint snapshot combining all component state.

    Represents a single point-in-time snapshot of training including:
    - Model weights (referenced by filename, stored separately)
    - Agent hyperparameters (may have evolved from initial config)
    - Environment configuration and reward parameters
    - Training progress and metadata
    - Recent loss history for trend analysis
    - Optional benchmark results and replay buffer

    The version field enables checkpoint format evolution. When adding new
    fields or changing structure, increment version and handle migration
    in loading logic.
    """

    version: int = 1
    experiment_name: str = ""
    epoch: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    model_weights_filename: str = ""
    agent_state: AgentCheckpointState | None = None
    env_state: EnvCheckpointState | None = None
    training_state: TrainingState | None = None
    loss_history: list[float] = field(default_factory=list)
    benchmark_results: dict | None = None
    replay_buffer_filename: str | None = None
    metadata: dict[str, any] = field(default_factory=dict)

    def get_identifier(self) -> str:
        """Generate unique checkpoint identifier.

        Returns string in format: {experiment_name}_epoch_{epoch}
        Used for checkpoint filenames and database lookups.
        """
        return f"{self.experiment_name}_epoch_{self.epoch}"
