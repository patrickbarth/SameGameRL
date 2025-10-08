# Checkpoint System Implementation Plan

## Implementation Status

**Overall Status**: 🟢 **CORE IMPLEMENTATION COMPLETE**

| Phase | Status | Files Created | Tests |
|-------|--------|---------------|-------|
| Phase 1: Core Data Structures | ✅ Complete | `checkpoint_data.py` | 13 tests passing |
| Phase 2: State Extraction | ✅ Complete | `checkpoint_state_extractor.py` | 6 tests passing |
| Phase 3: Pickle Repository | ✅ Complete | `pickle_checkpoint_repository.py` | 11 tests passing |
| Phase 4: Training Manager | ✅ Complete | `training_manager.py` (refactored) | 5 tests passing |
| Phase 5: Database Schema | ⏸️ Deferred | Not implemented | - |
| Phase 6: Database Repository | ⏸️ Deferred | Not implemented | - |
| Phase 7: Repository Factory | ⏸️ Deferred | Not implemented | - |
| Phase 8: Migration Scripts | ⏸️ Deferred | Not implemented | - |
| Phase 9: Documentation | ✅ Complete | Updated `CLAUDE.md` | - |

**Total Tests**: 48 passing (35 checkpoint + 13 training system)

**Key Achievements**:
- ✅ Full pickle-based checkpoint system operational
- ✅ Typed dataclasses with version field for format evolution
- ✅ Adapter pattern decouples checkpoint system from domain classes
- ✅ Single-directory storage (all checkpoint files colocated)
- ✅ TrainingManager refactored to OOP design (removed train() function)
- ✅ All tests passing after major refactoring

**Production Ready**: The pickle-based checkpoint system is fully functional and can be used for training. Database support (Phases 5-8) is optional and can be added later if needed.

---

## Executive Summary

**Status**: ✅ APPROVED WITH REFINEMENTS (Based on two independent expert reviews)

This plan implements a checkpoint tracking system for DQN training with dual storage (pickle + database) and full resumability. The design has been reviewed by two independent experts focusing on:
1. Clean Code principles and Python/ML best practices
2. Simplicity, extensibility, and implementation risks

**Key Refinements from Expert Reviews**:
- Use typed dataclasses instead of raw dicts for type safety
- Implement state extractor pattern to avoid coupling domain classes to checkpointing
- Decompose TrainingManager into focused services (SRP compliance)
- Add runtime database fallback for graceful degradation
- Add version field to CheckpointData for format evolution
- Remove unnecessary CheckpointManager abstraction layer

**Estimated Effort**: 13-17 days (adjusted from initial 8-12 days to incorporate quality improvements)

---

## Overview

Implement a comprehensive checkpoint tracking system for DQN training that captures:
- Model weights and optimizer state at different training points
- All hyperparameters (including those that may change during training)
- Reward function configuration (parameterized)
- Training loss history
- Benchmark performance data

The system must support:
- **Dual storage**: Pickle files (for remote GPU machines) + Database (for local analysis)
- **Resume training**: Load checkpoint and continue from that point
- **Cross-session training**: Train on one day, resume a week later
- **Bidirectional migration**: Sync between pickle and database storage

## Design Principles

1. **Separation of Concerns**: Each component (Agent, Env, TrainingManager) handles its own state
2. **Minimal Invasiveness**: Extend existing systems without breaking current workflows
3. **Simple but Extensible**: Core functionality only, designed for easy future expansion
4. **Test-Driven Development**: Write failing tests first, then implement

## Architecture

### Component Responsibilities

```
┌─────────────────────────────────────────────────────────────┐
│                     TrainingManager                          │
│  - Orchestrates training loop with checkpointing             │
│  - Collects state from Agent + Environment                   │
│  - Coordinates checkpoint creation and loading               │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─► DqnAgent
             │   └─► get_checkpoint_state() → hyperparameters
             │   └─► save() → model weights + hyperparameters
             │   └─► load() → restore weights + hyperparameters
             │
             ├─► SameGameEnv
             │   └─► get_checkpoint_state() → reward config
             │
             ├─► CheckpointManager
             │   └─► save_checkpoint() → persist to storage
             │   └─► load_checkpoint() → retrieve from storage
             │
             └─► Benchmark (optional)
                 └─► evaluate_agent() → performance metrics
```

### Data Model

#### CheckpointData (Core Data Structure)
```python
@dataclass
class CheckpointData:
    # Identity
    experiment_name: str
    epoch: int
    timestamp: datetime

    # Model reference
    model_weights_filename: str  # Agent handles save/load separately

    # Component states (from get_checkpoint_state())
    agent_state: dict  # {epsilon, gamma, tau, learning_rate, ...}
    env_state: dict    # {completion_reward, singles_reduction_weight, ...}
    training_state: dict  # {total_epochs, current_round, ...}

    # Metrics
    loss_history: list[float]  # Recent N losses

    # Optional
    benchmark_results: dict | None = None
    replay_buffer_filename: str | None = None
```

#### Database Schema (Optional Storage)
```
TrainingRun (one per experiment/model)
├── id
├── experiment_name
├── game_config_id (FK → GameConfig)
├── initial_config (JSON: starting hyperparameters)
├── created_at
└── checkpoints → [Checkpoint]

Checkpoint (many per TrainingRun)
├── id
├── training_run_id (FK → TrainingRun)
├── epoch_number
├── current_config (JSON: hyperparameters at this point)
├── model_weights_path (file path or blob)
├── optimizer_state_path (optional, can be in same file)
├── loss_history (JSON: recent losses)
├── created_at
└── benchmark_link → CheckpointBenchmark

CheckpointBenchmark (optional, one per Checkpoint)
├── id
├── checkpoint_id (FK → Checkpoint)
└── benchmark_metrics (JSON or FK to game_results)
```

**Key Design Decisions:**
- `initial_config` in TrainingRun: Snapshot of starting parameters
- `current_config` in Checkpoint: Parameters at this specific point (may have changed)
- JSON fields allow flexibility without schema migrations
- Supports parameter evolution (gamma, tau, reward weights changing during training)

## Implementation Phases

### Phase 1: Core Data Structures

**Files to Create:**
- `samegamerl/training/checkpoint_data.py`

**Classes:**
```python
@dataclass
class CheckpointData:
    """Complete checkpoint with state from all components."""
    experiment_name: str
    epoch: int
    timestamp: datetime
    model_weights_filename: str
    agent_state: dict
    env_state: dict
    training_state: dict
    loss_history: list[float]
    benchmark_results: dict | None = None
    replay_buffer_filename: str | None = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        ...

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointData":
        """Reconstruct from dict."""
        ...
```

**Tests to Write:**
- `samegamerl/tests/test_checkpoint_data.py`
  - ✓ Test dataclass creation with all required fields
  - ✓ Test to_dict() serialization
  - ✓ Test from_dict() deserialization
  - ✓ Test round-trip: data → dict → data
  - ✓ Test optional fields (benchmark_results, replay_buffer_filename)
  - ✓ Test datetime serialization/deserialization
  - ✓ Test validation of required fields

**TDD Workflow:**
```bash
# 1. Write failing test
pytest samegamerl/tests/test_checkpoint_data.py::test_checkpoint_creation -v
# 2. Implement CheckpointData
# 3. Test passes
```

---

### Phase 2: Component State Extraction

**Files to Modify:**
- `samegamerl/agents/dqn_agent.py`
- `samegamerl/environments/samegame_env.py`

**Agent Changes (DqnAgent):**
```python
def get_checkpoint_state(self) -> dict:
    """Return agent's current hyperparameters and training state.

    Used by TrainingManager to capture agent state for checkpointing.
    """
    return {
        'epsilon': self.epsilon,
        'epsilon_min': self.epsilon_min,
        'epsilon_decay': self.epsilon_decay,
        'learning_rate': self.learning_rate,
        'gamma': self.gamma,
        'tau': self.tau,
        'batch_size': self.batch_size,
        'replay_buffer_size': len(self.replay_buffer),
    }

def save(self, name: str | None = None):
    """Save model with hyperparameters for resumable training."""
    if not name:
        name = self.model_name

    model_path = self.models_dir / f"{name}.pth"
    torch.save(
        {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            # NEW: Include hyperparameters for resumable training
            "hyperparameters": self.get_checkpoint_state(),
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

def load(self, load_target=False, name: str | None = None):
    """Load model and optionally restore hyperparameters."""
    if not name:
        name = self.model_name

    model_path = self.models_dir / f"{name}.pth"
    checkpoint = torch.load(model_path, map_location=self.device)

    self.model.load_state_dict(checkpoint["model_state_dict"])
    self.opt.load_state_dict(checkpoint["optimizer_state_dict"])

    if load_target:
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
    else:
        self.target_model.load_state_dict(checkpoint["model_state_dict"])

    # NEW: Restore hyperparameters if present (backward compatible)
    if "hyperparameters" in checkpoint:
        self.epsilon = checkpoint["hyperparameters"].get("epsilon", self.epsilon)
        self.gamma = checkpoint["hyperparameters"].get("gamma", self.gamma)
        self.tau = checkpoint["hyperparameters"].get("tau", self.tau)
        self.learning_rate = checkpoint["hyperparameters"].get("learning_rate", self.learning_rate)
        # Note: Don't restore batch_size or min/decay (constructor params)

    self.model.train()
    self.target_model.eval()
    print(f"Model loaded from {model_path}")
```

**Environment Changes (SameGameEnv):**
```python
def get_checkpoint_state(self) -> dict:
    """Return environment's reward function configuration.

    Used by TrainingManager to capture env state for checkpointing.
    """
    return {
        'completion_reward': self.completion_reward,
        'partial_completion_base': self.partial_completion_base,
        'invalid_move_penalty': self.invalid_move_penalty,
        'singles_reduction_weight': self.singles_reduction_weight,
        'game_config': {
            'num_rows': self.config.num_rows,
            'num_cols': self.config.num_cols,
            'num_colors': self.config.num_colors,
        }
    }
```

**Tests to Write:**
- `samegamerl/tests/test_dqn_agent.py` (extend existing)
  - ✓ Test get_checkpoint_state() returns all hyperparameters
  - ✓ Test save() includes hyperparameters in checkpoint
  - ✓ Test load() restores hyperparameters
  - ✓ Test load() with old checkpoint (no hyperparameters) still works
  - ✓ Test epsilon preserved across save/load
  - ✓ Test gamma/tau preserved across save/load
  - ✓ Test backward compatibility

- `samegamerl/tests/test_samegame_env.py` (extend existing)
  - ✓ Test get_checkpoint_state() returns reward config
  - ✓ Test get_checkpoint_state() returns game config
  - ✓ Test all reward parameters included

**TDD Workflow:**
```bash
# 1. Write failing tests
pytest samegamerl/tests/test_dqn_agent.py::test_get_checkpoint_state -v
pytest samegamerl/tests/test_dqn_agent.py::test_save_includes_hyperparameters -v
# 2. Implement get_checkpoint_state() and modify save()
# 3. Tests pass
```

---

### Phase 3: Pickle Checkpoint Repository

**Files to Create:**
- `samegamerl/training/checkpoint_repository.py`

**Classes:**
```python
class PickleCheckpointRepository:
    """Handles persistence of checkpoints to pickle files."""

    def __init__(self, checkpoints_dir: Path | None = None):
        if checkpoints_dir is None:
            # Default: samegamerl/training/checkpoints/
            checkpoints_dir = Path(__file__).parent / "checkpoints"
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, checkpoint_data: CheckpointData) -> Path:
        """Save checkpoint to pickle file.

        Returns:
            Path to saved checkpoint file
        """
        filename = self._get_checkpoint_filename(
            checkpoint_data.experiment_name,
            checkpoint_data.epoch
        )
        filepath = self.checkpoints_dir / filename

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data.to_dict(), f)

        return filepath

    def load_checkpoint(self, checkpoint_id: str) -> CheckpointData | None:
        """Load checkpoint from pickle file.

        Args:
            checkpoint_id: Either full path or "experiment_name_epoch_XXXXX"

        Returns:
            CheckpointData or None if not found
        """
        filepath = self._resolve_checkpoint_path(checkpoint_id)

        if not filepath.exists():
            return None

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        return CheckpointData.from_dict(data)

    def list_checkpoints(self, experiment_name: str | None = None) -> list[str]:
        """List available checkpoints.

        Args:
            experiment_name: Filter by experiment, or None for all

        Returns:
            List of checkpoint IDs
        """
        pattern = f"{experiment_name}_epoch_*.pkl" if experiment_name else "*.pkl"
        checkpoints = sorted(self.checkpoints_dir.glob(pattern))
        return [cp.stem for cp in checkpoints]

    def _get_checkpoint_filename(self, experiment_name: str, epoch: int) -> str:
        """Generate checkpoint filename."""
        return f"{experiment_name}_epoch_{epoch:08d}.pkl"

    def _resolve_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Convert checkpoint ID to full path."""
        if "/" in checkpoint_id or checkpoint_id.endswith(".pkl"):
            return Path(checkpoint_id)

        if not checkpoint_id.endswith(".pkl"):
            checkpoint_id += ".pkl"

        return self.checkpoints_dir / checkpoint_id
```

**Tests to Write:**
- `samegamerl/tests/test_checkpoint_repository.py`
  - ✓ Test save_checkpoint creates file
  - ✓ Test load_checkpoint retrieves data correctly
  - ✓ Test round-trip: save → load → verify data integrity
  - ✓ Test list_checkpoints returns all checkpoints
  - ✓ Test list_checkpoints filters by experiment name
  - ✓ Test load_checkpoint with full path
  - ✓ Test load_checkpoint with experiment_epoch format
  - ✓ Test load_checkpoint returns None for missing file
  - ✓ Test checkpoint filename format
  - ✓ Test directory creation if not exists

**TDD Workflow:**
```bash
pytest samegamerl/tests/test_checkpoint_repository.py -v
```

---

### Phase 4: Training Manager

**Files to Create:**
- `samegamerl/training/training_manager.py`
- `samegamerl/training/checkpoint_manager.py`

**CheckpointManager (Helper):**
```python
class CheckpointManager:
    """Manages checkpoint persistence through repository abstraction."""

    def __init__(self, storage_type: str = "pickle", **kwargs):
        """
        Args:
            storage_type: "pickle" or "database"
            **kwargs: Passed to repository constructor
        """
        self.storage_type = storage_type

        if storage_type == "pickle":
            from samegamerl.training.checkpoint_repository import PickleCheckpointRepository
            self.repository = PickleCheckpointRepository(**kwargs)
        elif storage_type == "database":
            # Will implement in Phase 6
            raise NotImplementedError("Database storage not yet implemented")
        else:
            raise ValueError(f"Unknown storage_type: {storage_type}")

    def save_checkpoint(self, checkpoint_data: CheckpointData) -> str:
        """Save checkpoint and return checkpoint ID."""
        path = self.repository.save_checkpoint(checkpoint_data)
        return path.stem  # Return ID without extension

    def load_checkpoint(self, checkpoint_id: str) -> CheckpointData | None:
        """Load checkpoint by ID."""
        return self.repository.load_checkpoint(checkpoint_id)

    def list_checkpoints(self, experiment_name: str | None = None) -> list[str]:
        """List available checkpoints."""
        return self.repository.list_checkpoints(experiment_name)
```

**TrainingManager (Main Orchestrator):**
```python
from typing import Callable
from datetime import datetime
import pickle
from pathlib import Path

class TrainingManager:
    """Manages training lifecycle with checkpointing and evaluation."""

    def __init__(
        self,
        agent: DqnAgent,
        env: SameGameEnv,
        experiment_name: str,
        checkpoint_manager: CheckpointManager | None = None,
        starting_epoch: int = 0,
    ):
        self.agent = agent
        self.env = env
        self.experiment_name = experiment_name
        self.checkpoint_manager = checkpoint_manager
        self.total_epochs = starting_epoch
        self.current_round = 0
        self.start_time = datetime.now()

    def train_epoch(
        self,
        epochs: int,
        **train_params
    ) -> list[float]:
        """Single training run using existing train() function.

        Returns:
            Loss history from training
        """
        from samegamerl.training.train import train

        loss_history = train(self.agent, self.env, epochs=epochs, **train_params)
        self.total_epochs += epochs
        self.current_round += 1

        return loss_history

    def train_with_checkpoints(
        self,
        total_epochs: int,
        checkpoint_every: int,
        benchmark_every: int | None = None,
        benchmark_games: int = 500,
        save_replay_buffer: bool = False,
        **train_params
    ) -> dict:
        """Multi-round training with automatic checkpointing.

        Args:
            total_epochs: Total epochs to train
            checkpoint_every: Create checkpoint every N epochs
            benchmark_every: Run benchmark every N epochs (None = same as checkpoint)
            benchmark_games: Number of games for benchmarking
            save_replay_buffer: Whether to save replay buffer with checkpoint
            **train_params: Passed to train() function

        Returns:
            Dict with training progress data
        """
        if benchmark_every is None:
            benchmark_every = checkpoint_every

        num_rounds = total_epochs // checkpoint_every
        progress = {
            'epochs': [],
            'loss_history': [],
            'benchmark_results': []
        }

        for round_num in range(num_rounds):
            # Training
            loss_history = self.train_epoch(checkpoint_every, **train_params)
            progress['loss_history'].extend(loss_history)

            # Benchmarking (if configured)
            benchmark_results = None
            if round_num % (benchmark_every // checkpoint_every) == 0:
                benchmark_results = self._run_benchmark(benchmark_games)
                progress['benchmark_results'].append(benchmark_results)

            # Checkpointing
            if self.checkpoint_manager:
                self._create_checkpoint(
                    loss_history=loss_history,
                    benchmark_results=benchmark_results,
                    save_replay_buffer=save_replay_buffer
                )

            progress['epochs'].append(self.total_epochs)

        return progress

    def _run_benchmark(self, num_games: int) -> dict:
        """Run benchmark evaluation and return metrics."""
        from samegamerl.evaluation.benchmark_scripts import get_agent_performance

        results = get_agent_performance(self.agent, self.env.config, num_games)
        return results

    def _create_checkpoint(
        self,
        loss_history: list[float],
        benchmark_results: dict | None = None,
        save_replay_buffer: bool = False
    ) -> str:
        """Create checkpoint by collecting state from all components.

        Returns:
            Checkpoint ID
        """
        # 1. Save model weights (agent handles this)
        model_filename = f"{self.experiment_name}_epoch_{self.total_epochs:08d}"
        self.agent.save(name=model_filename)

        # 2. Optionally save replay buffer
        replay_buffer_filename = None
        if save_replay_buffer:
            replay_buffer_filename = f"{model_filename}_replay.pkl"
            replay_path = self.agent.models_dir / replay_buffer_filename
            with open(replay_path, 'wb') as f:
                pickle.dump(self.agent.replay_buffer, f)

        # 3. Collect state from all components
        checkpoint_data = CheckpointData(
            experiment_name=self.experiment_name,
            epoch=self.total_epochs,
            timestamp=datetime.now(),
            model_weights_filename=f"{model_filename}.pth",
            agent_state=self.agent.get_checkpoint_state(),
            env_state=self.env.get_checkpoint_state(),
            training_state=self.get_checkpoint_state(),
            loss_history=loss_history,
            benchmark_results=benchmark_results,
            replay_buffer_filename=replay_buffer_filename,
        )

        # 4. Persist via checkpoint manager
        checkpoint_id = self.checkpoint_manager.save_checkpoint(checkpoint_data)
        print(f"Checkpoint saved: {checkpoint_id} (epoch {self.total_epochs})")

        return checkpoint_id

    def get_checkpoint_state(self) -> dict:
        """Return training manager's state."""
        return {
            'total_epochs': self.total_epochs,
            'current_round': self.current_round,
            'start_time': self.start_time.isoformat(),
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_id: str,
        model_factory: Callable[[GameConfig], nn.Module],
        storage_type: str = "pickle",
        load_replay_buffer: bool = False,
    ) -> "TrainingManager":
        """Create TrainingManager by loading a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier (e.g., "experiment_epoch_00050000")
            model_factory: Function that creates model architecture
                          Example: lambda config: NeuralNetwork(config)
            storage_type: "pickle" or "database"
            load_replay_buffer: Whether to restore replay buffer

        Returns:
            TrainingManager instance ready to resume training

        Example:
            manager = TrainingManager.from_checkpoint(
                checkpoint_id="CNN_experiment_epoch_00050000",
                model_factory=lambda config: NeuralNetwork(config),
                load_replay_buffer=True
            )
            manager.train_with_checkpoints(total_epochs=50000, checkpoint_every=10000)
        """
        # 1. Load checkpoint metadata
        checkpoint_manager = CheckpointManager(storage_type=storage_type)
        checkpoint_data = checkpoint_manager.load_checkpoint(checkpoint_id)

        if checkpoint_data is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        # 2. Recreate environment from saved config
        env_config = checkpoint_data.env_state
        game_config = GameConfig(**env_config['game_config'])

        env = SameGameEnv(
            config=game_config,
            completion_reward=env_config['completion_reward'],
            partial_completion_base=env_config['partial_completion_base'],
            invalid_move_penalty=env_config['invalid_move_penalty'],
            singles_reduction_weight=env_config['singles_reduction_weight'],
        )

        # 3. Recreate agent with saved hyperparameters
        agent_config = checkpoint_data.agent_state
        agent = DqnAgent(
            model=model_factory(game_config),
            config=game_config,
            model_name=checkpoint_data.experiment_name,
            learning_rate=agent_config['learning_rate'],
            initial_epsilon=agent_config['epsilon'],  # Will be overridden by load()
            epsilon_decay=agent_config['epsilon_decay'],
            final_epsilon=agent_config['epsilon_min'],
            batch_size=agent_config['batch_size'],
            gamma=agent_config['gamma'],
            tau=agent_config['tau'],
        )

        # 4. Load model weights and hyperparameters
        model_name = checkpoint_data.model_weights_filename.replace('.pth', '')
        agent.load(name=model_name)

        # 5. Optionally load replay buffer
        if load_replay_buffer and checkpoint_data.replay_buffer_filename:
            replay_path = agent.models_dir / checkpoint_data.replay_buffer_filename
            if replay_path.exists():
                with open(replay_path, 'rb') as f:
                    agent.replay_buffer = pickle.load(f)
                print(f"Replay buffer loaded: {len(agent.replay_buffer)} experiences")

        # 6. Create manager with restored state
        manager = cls(
            agent=agent,
            env=env,
            experiment_name=checkpoint_data.experiment_name,
            checkpoint_manager=checkpoint_manager,
            starting_epoch=checkpoint_data.epoch,
        )

        # Restore training state
        training_state = checkpoint_data.training_state
        manager.current_round = training_state.get('current_round', 0)
        if 'start_time' in training_state:
            manager.start_time = datetime.fromisoformat(training_state['start_time'])

        print(f"Resumed from checkpoint: {checkpoint_id} (epoch {checkpoint_data.epoch})")

        return manager
```

**Tests to Write:**
- `samegamerl/tests/test_checkpoint_manager.py`
  - ✓ Test save_checkpoint with pickle storage
  - ✓ Test load_checkpoint retrieves correct data
  - ✓ Test list_checkpoints
  - ✓ Test invalid storage_type raises error

- `samegamerl/tests/test_training_manager.py`
  - ✓ Test train_epoch increments total_epochs
  - ✓ Test train_epoch returns loss history
  - ✓ Test train_with_checkpoints creates checkpoints
  - ✓ Test checkpoint includes agent state
  - ✓ Test checkpoint includes env state
  - ✓ Test checkpoint includes training state
  - ✓ Test checkpoint includes benchmark results
  - ✓ Test checkpoint saves replay buffer when requested
  - ✓ Test from_checkpoint recreates environment correctly
  - ✓ Test from_checkpoint recreates agent with correct hyperparameters
  - ✓ Test from_checkpoint restores training state
  - ✓ Test from_checkpoint loads replay buffer when requested
  - ✓ Test resuming training continues from correct epoch
  - ✓ Test from_checkpoint raises error for missing checkpoint

**TDD Workflow:**
```bash
pytest samegamerl/tests/test_training_manager.py -v
```

---

### Phase 5: Database Schema (Optional Storage)

**Files to Create:**
- `alembic/versions/XXX_add_checkpoint_tables.py` (migration)

**Files to Modify:**
- `samegamerl/database/models.py`

**New Database Models:**
```python
class TrainingRun(Base):
    """Represents a training session for a specific experiment."""
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_name = Column(String(255), nullable=False, index=True)
    game_config_id = Column(Integer, ForeignKey("game_configs.id"), nullable=False)

    # Initial hyperparameters (snapshot at training start)
    initial_config = Column(JSON, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    config = relationship("GameConfig")
    checkpoints = relationship("Checkpoint", back_populates="training_run", cascade="all, delete-orphan")

    __table_args__ = (Index("idx_training_runs_experiment", "experiment_name"),)

    def __repr__(self) -> str:
        return f"<TrainingRun({self.experiment_name}, {len(self.checkpoints)} checkpoints)>"


class Checkpoint(Base):
    """Represents a specific checkpoint in a training run."""
    __tablename__ = "checkpoints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False)
    epoch_number = Column(Integer, nullable=False)

    # Current hyperparameters (may have changed from initial)
    current_config = Column(JSON, nullable=False)

    # Model state references
    model_weights_path = Column(String(500), nullable=False)
    optimizer_state_path = Column(String(500))  # Optional: might be in same file as model
    replay_buffer_path = Column(String(500))  # Optional

    # Training metrics
    loss_history = Column(JSON)  # List of recent losses

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    training_run = relationship("TrainingRun", back_populates="checkpoints")
    benchmark_link = relationship("CheckpointBenchmark", uselist=False, back_populates="checkpoint", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("training_run_id", "epoch_number"),
        Index("idx_checkpoints_epoch", "training_run_id", "epoch_number"),
    )

    def __repr__(self) -> str:
        return f"<Checkpoint(run={self.training_run_id}, epoch={self.epoch_number})>"


class CheckpointBenchmark(Base):
    """Links a checkpoint to its benchmark results."""
    __tablename__ = "checkpoint_benchmarks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    checkpoint_id = Column(Integer, ForeignKey("checkpoints.id"), unique=True, nullable=False)

    # Benchmark metrics (JSON for flexibility)
    # Could also link to existing game_results table if using database benchmarks
    benchmark_metrics = Column(JSON, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    checkpoint = relationship("Checkpoint", back_populates="benchmark_link")

    def __repr__(self) -> str:
        return f"<CheckpointBenchmark(checkpoint={self.checkpoint_id})>"
```

**Alembic Migration:**
```python
"""Add checkpoint tables

Revision ID: XXX
Revises: YYY
Create Date: 2025-XX-XX
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    # Create training_runs table
    op.create_table(
        'training_runs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('experiment_name', sa.String(255), nullable=False),
        sa.Column('game_config_id', sa.Integer(), nullable=False),
        sa.Column('initial_config', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['game_config_id'], ['game_configs.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_training_runs_experiment', 'training_runs', ['experiment_name'])

    # Create checkpoints table
    op.create_table(
        'checkpoints',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('training_run_id', sa.Integer(), nullable=False),
        sa.Column('epoch_number', sa.Integer(), nullable=False),
        sa.Column('current_config', sa.JSON(), nullable=False),
        sa.Column('model_weights_path', sa.String(500), nullable=False),
        sa.Column('optimizer_state_path', sa.String(500), nullable=True),
        sa.Column('replay_buffer_path', sa.String(500), nullable=True),
        sa.Column('loss_history', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['training_run_id'], ['training_runs.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('training_run_id', 'epoch_number')
    )
    op.create_index('idx_checkpoints_epoch', 'checkpoints', ['training_run_id', 'epoch_number'])

    # Create checkpoint_benchmarks table
    op.create_table(
        'checkpoint_benchmarks',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('checkpoint_id', sa.Integer(), nullable=False),
        sa.Column('benchmark_metrics', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['checkpoint_id'], ['checkpoints.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('checkpoint_id')
    )

def downgrade():
    op.drop_table('checkpoint_benchmarks')
    op.drop_table('checkpoints')
    op.drop_table('training_runs')
```

**Tests to Write:**
- `samegamerl/tests/test_checkpoint_database_models.py`
  - ✓ Test creating TrainingRun with initial config
  - ✓ Test creating Checkpoint linked to TrainingRun
  - ✓ Test creating CheckpointBenchmark linked to Checkpoint
  - ✓ Test unique constraint on (training_run_id, epoch_number)
  - ✓ Test cascade delete (deleting TrainingRun deletes Checkpoints)
  - ✓ Test JSON field serialization for configs
  - ✓ Test querying checkpoints by experiment name
  - ✓ Test querying checkpoints by epoch range

**TDD Workflow:**
```bash
# 1. Create migration
alembic revision -m "add checkpoint tables"
# 2. Write migration upgrade/downgrade
# 3. Apply migration
alembic upgrade head
# 4. Write tests
pytest samegamerl/tests/test_checkpoint_database_models.py -v
```

---

### Phase 6: Database Checkpoint Repository

**Files to Create:**
- `samegamerl/training/database_checkpoint_repository.py`

**Classes:**
```python
from typing import Optional
from samegamerl.database.repository import DatabaseRepository
from samegamerl.database.models import TrainingRun, Checkpoint, CheckpointBenchmark
from samegamerl.training.checkpoint_data import CheckpointData
from samegamerl.game.game_config import GameConfig

class DatabaseCheckpointRepository:
    """Handles persistence of checkpoints to database."""

    def __init__(self):
        """Initialize with database connection."""
        from samegamerl.database.availability import is_database_available

        if not is_database_available():
            raise RuntimeError(
                "Database dependencies not available. "
                "Install with: poetry install -E database"
            )

        self.db = DatabaseRepository()

    def save_checkpoint(self, checkpoint_data: CheckpointData) -> str:
        """Save checkpoint to database.

        Returns:
            Checkpoint ID (string format: "experiment_epoch_XXXXX")
        """
        # 1. Find or create game config
        env_state = checkpoint_data.env_state
        game_config_dict = env_state['game_config']

        db_config = self.db.game_configs.find_or_create(
            num_rows=game_config_dict['num_rows'],
            num_cols=game_config_dict['num_cols'],
            num_colors=game_config_dict['num_colors'],
        )

        # 2. Find or create training run
        training_run = self._find_or_create_training_run(
            experiment_name=checkpoint_data.experiment_name,
            game_config_id=db_config.id,
            initial_config=self._build_initial_config(checkpoint_data),
        )

        # 3. Create checkpoint record
        current_config = self._build_current_config(checkpoint_data)

        checkpoint = Checkpoint(
            training_run_id=training_run.id,
            epoch_number=checkpoint_data.epoch,
            current_config=current_config,
            model_weights_path=checkpoint_data.model_weights_filename,
            replay_buffer_path=checkpoint_data.replay_buffer_filename,
            loss_history=checkpoint_data.loss_history,
        )

        self.db.session.add(checkpoint)
        self.db.session.flush()

        # 4. Create benchmark link if results provided
        if checkpoint_data.benchmark_results:
            benchmark = CheckpointBenchmark(
                checkpoint_id=checkpoint.id,
                benchmark_metrics=checkpoint_data.benchmark_results,
            )
            self.db.session.add(benchmark)

        self.db.commit()

        return f"{checkpoint_data.experiment_name}_epoch_{checkpoint_data.epoch:08d}"

    def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointData]:
        """Load checkpoint from database.

        Args:
            checkpoint_id: Format "experiment_epoch_XXXXX" or numeric ID

        Returns:
            CheckpointData or None if not found
        """
        # Parse checkpoint_id
        if checkpoint_id.isdigit():
            # Numeric ID
            checkpoint = self.db.session.query(Checkpoint).filter_by(id=int(checkpoint_id)).first()
        else:
            # experiment_epoch_XXXXX format
            parts = checkpoint_id.split('_epoch_')
            if len(parts) != 2:
                return None

            experiment_name = parts[0]
            epoch = int(parts[1])

            checkpoint = (
                self.db.session.query(Checkpoint)
                .join(TrainingRun)
                .filter(
                    TrainingRun.experiment_name == experiment_name,
                    Checkpoint.epoch_number == epoch
                )
                .first()
            )

        if not checkpoint:
            return None

        # Reconstruct CheckpointData
        training_run = checkpoint.training_run

        # Get game config
        game_config = training_run.config
        game_config_dict = {
            'num_rows': game_config.num_rows,
            'num_cols': game_config.num_cols,
            'num_colors': game_config.num_colors,
        }

        # Extract configs
        current_config = checkpoint.current_config
        agent_state = {
            'epsilon': current_config['epsilon'],
            'epsilon_min': current_config['epsilon_min'],
            'epsilon_decay': current_config['epsilon_decay'],
            'learning_rate': current_config['learning_rate'],
            'gamma': current_config['gamma'],
            'tau': current_config['tau'],
            'batch_size': current_config['batch_size'],
            'replay_buffer_size': current_config.get('replay_buffer_size', 0),
        }

        env_state = {
            'game_config': game_config_dict,
            'completion_reward': current_config['completion_reward'],
            'partial_completion_base': current_config['partial_completion_base'],
            'invalid_move_penalty': current_config['invalid_move_penalty'],
            'singles_reduction_weight': current_config['singles_reduction_weight'],
        }

        training_state = {
            'total_epochs': checkpoint.epoch_number,
            'current_round': current_config.get('current_round', 0),
            'start_time': current_config.get('start_time', checkpoint.created_at.isoformat()),
        }

        # Get benchmark results if available
        benchmark_results = None
        if checkpoint.benchmark_link:
            benchmark_results = checkpoint.benchmark_link.benchmark_metrics

        return CheckpointData(
            experiment_name=training_run.experiment_name,
            epoch=checkpoint.epoch_number,
            timestamp=checkpoint.created_at,
            model_weights_filename=checkpoint.model_weights_path,
            agent_state=agent_state,
            env_state=env_state,
            training_state=training_state,
            loss_history=checkpoint.loss_history or [],
            benchmark_results=benchmark_results,
            replay_buffer_filename=checkpoint.replay_buffer_path,
        )

    def list_checkpoints(self, experiment_name: str | None = None) -> list[str]:
        """List available checkpoints.

        Args:
            experiment_name: Filter by experiment, or None for all

        Returns:
            List of checkpoint IDs in format "experiment_epoch_XXXXX"
        """
        query = (
            self.db.session.query(Checkpoint, TrainingRun)
            .join(TrainingRun)
            .order_by(TrainingRun.experiment_name, Checkpoint.epoch_number)
        )

        if experiment_name:
            query = query.filter(TrainingRun.experiment_name == experiment_name)

        results = query.all()

        return [
            f"{run.experiment_name}_epoch_{checkpoint.epoch_number:08d}"
            for checkpoint, run in results
        ]

    def _find_or_create_training_run(
        self,
        experiment_name: str,
        game_config_id: int,
        initial_config: dict
    ) -> TrainingRun:
        """Find existing training run or create new one."""
        training_run = (
            self.db.session.query(TrainingRun)
            .filter_by(experiment_name=experiment_name, game_config_id=game_config_id)
            .first()
        )

        if not training_run:
            training_run = TrainingRun(
                experiment_name=experiment_name,
                game_config_id=game_config_id,
                initial_config=initial_config,
            )
            self.db.session.add(training_run)
            self.db.session.flush()

        return training_run

    def _build_initial_config(self, checkpoint_data: CheckpointData) -> dict:
        """Build initial config from checkpoint data."""
        # For the first checkpoint, this is the initial config
        # For subsequent checkpoints, we use the existing training run's config
        return {
            **checkpoint_data.agent_state,
            **checkpoint_data.env_state,
        }

    def _build_current_config(self, checkpoint_data: CheckpointData) -> dict:
        """Build current config from checkpoint data."""
        return {
            **checkpoint_data.agent_state,
            **checkpoint_data.env_state,
            **checkpoint_data.training_state,
        }
```

**Tests to Write:**
- `samegamerl/tests/test_database_checkpoint_repository.py`
  - ✓ Test save_checkpoint creates database records
  - ✓ Test load_checkpoint retrieves correct data
  - ✓ Test round-trip: save → load → verify integrity
  - ✓ Test list_checkpoints returns all
  - ✓ Test list_checkpoints filters by experiment
  - ✓ Test load_checkpoint by numeric ID
  - ✓ Test load_checkpoint by experiment_epoch format
  - ✓ Test load_checkpoint returns None for missing
  - ✓ Test multiple checkpoints for same experiment
  - ✓ Test benchmark results saved and loaded
  - ✓ Test optional dependencies gracefully handled

**TDD Workflow:**
```bash
pytest samegamerl/tests/test_database_checkpoint_repository.py -v
```

---

### Phase 7: Repository Factory

**Files to Create:**
- `samegamerl/training/checkpoint_repository_factory.py`

**Classes:**
```python
from samegamerl.database.availability import is_database_available

class CheckpointRepositoryFactory:
    """Factory for creating appropriate checkpoint repository."""

    @staticmethod
    def create(storage_type: str = "auto", **kwargs):
        """Create checkpoint repository based on storage type.

        Args:
            storage_type: "auto", "pickle", or "database"
                - "auto": Use database if available, otherwise pickle
                - "pickle": Force pickle storage
                - "database": Force database (raises error if unavailable)
            **kwargs: Passed to repository constructor

        Returns:
            Repository instance (PickleCheckpointRepository or DatabaseCheckpointRepository)
        """
        if storage_type == "auto":
            storage_type = "database" if is_database_available() else "pickle"

        if storage_type == "pickle":
            from samegamerl.training.checkpoint_repository import PickleCheckpointRepository
            return PickleCheckpointRepository(**kwargs)

        elif storage_type == "database":
            if not is_database_available():
                raise RuntimeError(
                    "Database dependencies not available. "
                    "Install with: poetry install -E database"
                )
            from samegamerl.training.database_checkpoint_repository import DatabaseCheckpointRepository
            return DatabaseCheckpointRepository(**kwargs)

        else:
            raise ValueError(
                f"Invalid storage_type '{storage_type}'. "
                "Must be 'auto', 'pickle', or 'database'"
            )
```

**Update CheckpointManager:**
```python
class CheckpointManager:
    def __init__(self, storage_type: str = "auto", **kwargs):
        """
        Args:
            storage_type: "auto", "pickle", or "database"
            **kwargs: Passed to repository constructor
        """
        from samegamerl.training.checkpoint_repository_factory import CheckpointRepositoryFactory
        self.repository = CheckpointRepositoryFactory.create(storage_type, **kwargs)
        self.storage_type = storage_type
```

**Tests to Write:**
- `samegamerl/tests/test_checkpoint_repository_factory.py`
  - ✓ Test factory creates pickle repo when requested
  - ✓ Test factory creates database repo when available
  - ✓ Test factory auto-selects based on availability
  - ✓ Test factory raises error for invalid type
  - ✓ Test factory raises error for database when unavailable

**TDD Workflow:**
```bash
pytest samegamerl/tests/test_checkpoint_repository_factory.py -v
```

---

### Phase 8: Migration Scripts

**Files to Create:**
- `scripts/migrate_checkpoints_to_db.py`
- `scripts/export_checkpoints_to_pickle.py`

**Pickle → Database Migration:**
```python
#!/usr/bin/env python3
"""Migrate checkpoint pickle files to database.

Usage:
    python scripts/migrate_checkpoints_to_db.py [--checkpoints-dir PATH]
"""

import argparse
from pathlib import Path
from samegamerl.training.checkpoint_repository import PickleCheckpointRepository
from samegamerl.training.database_checkpoint_repository import DatabaseCheckpointRepository

def migrate_checkpoints(checkpoints_dir: Path | None = None):
    """Migrate all pickle checkpoints to database."""

    # Initialize repositories
    pickle_repo = PickleCheckpointRepository(checkpoints_dir=checkpoints_dir)
    db_repo = DatabaseCheckpointRepository()

    # Get all checkpoints
    checkpoint_ids = pickle_repo.list_checkpoints()

    print(f"Found {len(checkpoint_ids)} checkpoints to migrate")

    migrated = 0
    skipped = 0
    errors = 0

    for checkpoint_id in checkpoint_ids:
        try:
            # Load from pickle
            checkpoint_data = pickle_repo.load_checkpoint(checkpoint_id)

            if checkpoint_data is None:
                print(f"  ⚠ Skipped {checkpoint_id} (failed to load)")
                skipped += 1
                continue

            # Check if already in database
            existing = db_repo.load_checkpoint(checkpoint_id)
            if existing:
                print(f"  → Skipped {checkpoint_id} (already in database)")
                skipped += 1
                continue

            # Save to database
            db_repo.save_checkpoint(checkpoint_data)
            print(f"  ✓ Migrated {checkpoint_id}")
            migrated += 1

        except Exception as e:
            print(f"  ✗ Error migrating {checkpoint_id}: {e}")
            errors += 1

    print(f"\nMigration complete:")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate checkpoints from pickle to database")
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        help="Path to checkpoints directory (default: samegamerl/training/checkpoints)",
    )

    args = parser.parse_args()
    migrate_checkpoints(args.checkpoints_dir)
```

**Database → Pickle Export:**
```python
#!/usr/bin/env python3
"""Export checkpoints from database to pickle files.

Useful for transferring checkpoints to remote GPU machines.

Usage:
    python scripts/export_checkpoints_to_pickle.py [--experiment NAME] [--output-dir PATH]
"""

import argparse
from pathlib import Path
from samegamerl.training.checkpoint_repository import PickleCheckpointRepository
from samegamerl.training.database_checkpoint_repository import DatabaseCheckpointRepository

def export_checkpoints(
    experiment_name: str | None = None,
    output_dir: Path | None = None
):
    """Export checkpoints from database to pickle files."""

    # Initialize repositories
    db_repo = DatabaseCheckpointRepository()
    pickle_repo = PickleCheckpointRepository(checkpoints_dir=output_dir)

    # Get checkpoints to export
    checkpoint_ids = db_repo.list_checkpoints(experiment_name=experiment_name)

    if experiment_name:
        print(f"Found {len(checkpoint_ids)} checkpoints for experiment '{experiment_name}'")
    else:
        print(f"Found {len(checkpoint_ids)} checkpoints across all experiments")

    exported = 0
    skipped = 0
    errors = 0

    for checkpoint_id in checkpoint_ids:
        try:
            # Load from database
            checkpoint_data = db_repo.load_checkpoint(checkpoint_id)

            if checkpoint_data is None:
                print(f"  ⚠ Skipped {checkpoint_id} (failed to load)")
                skipped += 1
                continue

            # Check if already exists in pickle
            existing = pickle_repo.load_checkpoint(checkpoint_id)
            if existing:
                print(f"  → Skipped {checkpoint_id} (already exists)")
                skipped += 1
                continue

            # Save to pickle
            pickle_repo.save_checkpoint(checkpoint_data)
            print(f"  ✓ Exported {checkpoint_id}")
            exported += 1

        except Exception as e:
            print(f"  ✗ Error exporting {checkpoint_id}: {e}")
            errors += 1

    print(f"\nExport complete:")
    print(f"  Exported: {exported}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")

    if output_dir:
        print(f"\nPickle files saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export checkpoints from database to pickle")
    parser.add_argument(
        "--experiment",
        type=str,
        help="Export only checkpoints for this experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for pickle files (default: samegamerl/training/checkpoints)",
    )

    args = parser.parse_args()
    export_checkpoints(args.experiment, args.output_dir)
```

**Tests to Write:**
- `samegamerl/tests/test_checkpoint_migration.py`
  - ✓ Test migrate_checkpoints transfers data correctly
  - ✓ Test export_checkpoints transfers data correctly
  - ✓ Test round-trip: pickle → db → pickle preserves data
  - ✓ Test migration skips existing checkpoints
  - ✓ Test export filters by experiment name
  - ✓ Test error handling for corrupt data

**TDD Workflow:**
```bash
pytest samegamerl/tests/test_checkpoint_migration.py -v
```

---

### Phase 9: Documentation

**Files to Update:**
- `CLAUDE.md`
- Create or update `ARCHITECTURE.md`

**CLAUDE.md additions:**
```markdown
## Checkpoint System for Training Tracking

The checkpoint system tracks model training progress with full context for resumable training.

### Basic Usage

```python
from samegamerl.training.training_manager import TrainingManager
from samegamerl.training.checkpoint_manager import CheckpointManager

# Create checkpoint manager
checkpoint_mgr = CheckpointManager(storage_type="pickle")  # or "database" or "auto"

# Create training manager with checkpointing
manager = TrainingManager(
    agent=agent,
    env=env,
    experiment_name="CNN_experiment",
    checkpoint_manager=checkpoint_mgr
)

# Train with automatic checkpointing
manager.train_with_checkpoints(
    total_epochs=100000,
    checkpoint_every=10000,
    benchmark_every=10000,
    benchmark_games=500,
    save_replay_buffer=True,
    # ... other train() parameters
)
```

### Resuming Training

```python
# Resume from checkpoint (one week later)
manager = TrainingManager.from_checkpoint(
    checkpoint_id="CNN_experiment_epoch_00050000",
    model_factory=lambda config: NeuralNetwork(config),
    storage_type="pickle",
    load_replay_buffer=True
)

# Continue training
manager.train_with_checkpoints(total_epochs=50000, checkpoint_every=10000)
```

### What's Captured in Checkpoints

- **Model state**: Weights, optimizer state, target network
- **Hyperparameters**: Epsilon, gamma, tau, learning rate (current values)
- **Reward configuration**: All SameGameEnv reward parameters
- **Training metrics**: Loss history, epoch count
- **Benchmark results**: Performance evaluation (optional)
- **Replay buffer**: Experience replay (optional, saves separately)

### Storage Options

- **Pickle** (default): File-based, works on remote GPU machines
- **Database**: PostgreSQL-based, enables querying and analysis
- **Auto**: Uses database if available, otherwise pickle

### Migration Between Storage Types

```bash
# Export from database to pickle (for remote training)
python scripts/export_checkpoints_to_pickle.py --experiment CNN_experiment

# Import from pickle to database (for analysis)
python scripts/migrate_checkpoints_to_db.py
```

### Design Principles

- Each component (Agent, Environment) exposes its state via `get_checkpoint_state()`
- TrainingManager orchestrates checkpoint creation without tight coupling
- Model architecture must be provided when loading (via `model_factory`)
- Checkpoints are fully self-contained for resumable training
```

**ARCHITECTURE.md additions:**
```markdown
## Checkpoint System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     TrainingManager                          │
│  - Orchestrates training with checkpointing                  │
│  - Collects state from all components                        │
│  - Handles checkpoint creation and loading                   │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─► DqnAgent.get_checkpoint_state()
             │   Returns: {epsilon, gamma, tau, lr, ...}
             │
             ├─► SameGameEnv.get_checkpoint_state()
             │   Returns: {reward params, game config, ...}
             │
             ├─► CheckpointManager.save_checkpoint(data)
             │   └─► CheckpointRepository (Pickle or Database)
             │
             └─► Benchmark.evaluate_agent() [optional]
                 Returns: Performance metrics
```

### Data Flow

**Creating a Checkpoint:**
1. TrainingManager calls `train()` → gets loss history
2. TrainingManager calls `Benchmark.evaluate_agent()` → gets metrics
3. TrainingManager calls `agent.get_checkpoint_state()` → gets hyperparameters
4. TrainingManager calls `env.get_checkpoint_state()` → gets reward config
5. TrainingManager calls `agent.save()` → saves model weights
6. TrainingManager assembles CheckpointData
7. TrainingManager calls `CheckpointManager.save_checkpoint()`
8. CheckpointRepository persists to storage (pickle or database)

**Loading a Checkpoint:**
1. User calls `TrainingManager.from_checkpoint(id, model_factory)`
2. TrainingManager loads CheckpointData from repository
3. TrainingManager recreates Environment with saved reward config
4. TrainingManager recreates Agent with saved hyperparameters
5. Agent loads model weights via `load()`
6. Agent's `load()` restores epsilon, gamma, tau from checkpoint
7. TrainingManager restores training state (epoch count, etc.)
8. Optional: Load replay buffer if requested
9. Returns ready-to-train TrainingManager instance

### Database Schema

```
TrainingRun
├── experiment_name
├── game_config_id → GameConfig
├── initial_config (JSON)
└── checkpoints → [Checkpoint]

Checkpoint
├── training_run_id → TrainingRun
├── epoch_number
├── current_config (JSON)  ← May differ from initial_config
├── model_weights_path
├── loss_history (JSON)
└── benchmark_link → CheckpointBenchmark

CheckpointBenchmark
├── checkpoint_id → Checkpoint
└── benchmark_metrics (JSON)
```

**Configuration Evolution:**
- `TrainingRun.initial_config`: Hyperparameters at start
- `Checkpoint.current_config`: Hyperparameters at this checkpoint
- Enables tracking parameter changes (gamma, tau, reward weights)
- Query example: `SELECT epoch_number, current_config->>'gamma' FROM checkpoints`

### Storage Patterns

**Pickle Storage:**
- Files: `samegamerl/training/checkpoints/experiment_epoch_XXXXX.pkl`
- Contains: Full CheckpointData serialized
- Model weights: Separate `.pth` file in `samegamerl/models/`
- Replay buffer: Separate `_replay.pkl` file (optional)

**Database Storage:**
- CheckpointData split across normalized tables
- Model weights: File path reference (not stored in DB)
- Enables SQL queries for analysis and comparison
- Supports optional dependency pattern

### Key Design Decisions

1. **Model Architecture Not Serialized**: User provides `model_factory` function
   - Reason: Python code can't be reliably serialized
   - Pattern: Common in ML frameworks (Keras, PyTorch Lightning)

2. **Dual Storage Support**: Pickle and Database
   - Reason: Remote GPU machines may not have database access
   - Pattern: Repository abstraction with factory

3. **Hyperparameters in Agent Save**: Agent's `save()` includes hyperparameters
   - Reason: Enables simple resume without full checkpoint system
   - Pattern: Backward compatible (old saves still work)

4. **Separate Replay Buffer**: Not in main checkpoint file
   - Reason: Large file size, optional for resume
   - Pattern: Reference by filename, load if requested

5. **JSON Config Storage**: Database uses JSON for configs
   - Reason: Schema-less flexibility for adding parameters
   - Pattern: No migrations needed when adding hyperparameters
```

---

## Implementation Summary

### Files Created (18 new files)
1. `samegamerl/training/checkpoint_data.py`
2. `samegamerl/training/checkpoint_repository.py`
3. `samegamerl/training/database_checkpoint_repository.py`
4. `samegamerl/training/checkpoint_repository_factory.py`
5. `samegamerl/training/checkpoint_manager.py`
6. `samegamerl/training/training_manager.py`
7. `alembic/versions/XXX_add_checkpoint_tables.py`
8. `scripts/migrate_checkpoints_to_db.py`
9. `scripts/export_checkpoints_to_pickle.py`
10. `samegamerl/tests/test_checkpoint_data.py`
11. `samegamerl/tests/test_checkpoint_repository.py`
12. `samegamerl/tests/test_checkpoint_manager.py`
13. `samegamerl/tests/test_training_manager.py`
14. `samegamerl/tests/test_checkpoint_database_models.py`
15. `samegamerl/tests/test_database_checkpoint_repository.py`
16. `samegamerl/tests/test_checkpoint_repository_factory.py`
17. `samegamerl/tests/test_checkpoint_migration.py`
18. `CHECKPOINT_SYSTEM_PLAN.md` (this file)

### Files Modified (4 files)
1. `samegamerl/agents/dqn_agent.py` - Add `get_checkpoint_state()`, extend `save()`/`load()`
2. `samegamerl/environments/samegame_env.py` - Add `get_checkpoint_state()`
3. `samegamerl/database/models.py` - Add TrainingRun, Checkpoint, CheckpointBenchmark models
4. `CLAUDE.md` - Add checkpoint system documentation

### Test Coverage
- **17 test files** covering all components
- **~120+ test cases** across all phases
- Focus on: data integrity, round-trip serialization, resumable training, migration

### Dependencies
- **Core**: No new dependencies (uses existing pickle, torch, datetime)
- **Database** (optional): Already available via existing database dependencies

### Estimated Effort
- **Phase 1-3**: 2-3 days (Core data + component changes)
- **Phase 4**: 2-3 days (TrainingManager with loading)
- **Phase 5-6**: 2-3 days (Database support)
- **Phase 7-8**: 1-2 days (Factory + migration)
- **Phase 9**: 1 day (Documentation)
- **Total**: 8-12 days with TDD approach

### Success Criteria
- [ ] Can create checkpoint during training
- [ ] Can resume training from checkpoint (epsilon preserved)
- [ ] Can change reward parameters and track in checkpoint
- [ ] Can migrate between pickle and database storage
- [ ] All tests pass
- [ ] Documentation complete
- [ ] Backward compatible with existing code

## Risk Mitigation

### Potential Issues

1. **Model architecture mismatch on load**
   - Mitigation: Require explicit model_factory, fail fast with clear error
   - Test: Verify error message when architecture doesn't match

2. **Checkpoint file size with replay buffer**
   - Mitigation: Separate replay buffer file, optional loading
   - Test: Verify checkpoint size reasonable without buffer

3. **Database schema evolution**
   - Mitigation: Use JSON for configs, no migration needed for new params
   - Test: Verify loading old checkpoints with missing params

4. **Hyperparameter restoration edge cases**
   - Mitigation: Use `.get()` with defaults, backward compatible
   - Test: Load old checkpoints without hyperparameters

5. **Training state inconsistency**
   - Mitigation: Atomic checkpoint creation, verify before save
   - Test: Verify training continues correctly after resume

## Open Questions for Discussion

1. **Loss history window size**: How many recent losses to store? (100? 1000?)
2. **Checkpoint retention**: Auto-delete old checkpoints? Keep every Nth?
3. **Benchmark integration**: Always run on checkpoint? Make it truly optional?
4. **Model weights storage**: Keep in separate files or embed in checkpoint?
5. **Compression**: Compress large replay buffers?

## Next Steps

1. Review this plan with stakeholders
2. Confirm design decisions and open questions
3. Begin Phase 1 implementation with TDD
4. Iterate based on feedback from early phases