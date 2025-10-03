"""Tests for checkpoint data structures."""

from datetime import datetime
from pathlib import Path

import pytest

from samegamerl.game.game_config import GameFactory
from samegamerl.training.checkpoint_data import (
    AgentCheckpointState,
    CheckpointData,
    EnvCheckpointState,
    TrainingState,
)


class TestAgentCheckpointState:
    """Tests for AgentCheckpointState dataclass."""

    def test_basic_initialization(self):
        """Test creating AgentCheckpointState with all required fields."""
        state = AgentCheckpointState(
            epsilon=0.5,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            learning_rate=0.001,
            gamma=0.99,
            tau=0.005,
            batch_size=64,
            replay_buffer_size=10000,
        )

        assert state.epsilon == 0.5
        assert state.epsilon_min == 0.01
        assert state.epsilon_decay == 0.995
        assert state.learning_rate == 0.001
        assert state.gamma == 0.99
        assert state.tau == 0.005
        assert state.batch_size == 64
        assert state.replay_buffer_size == 10000

    def test_type_validation(self):
        """Test that type hints are enforced."""
        # This should work with floats
        state = AgentCheckpointState(
            epsilon=0.5,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            learning_rate=0.001,
            gamma=0.99,
            tau=0.005,
            batch_size=64,
            replay_buffer_size=10000,
        )
        assert isinstance(state.epsilon, float)
        assert isinstance(state.batch_size, int)


class TestEnvCheckpointState:
    """Tests for EnvCheckpointState dataclass."""

    def test_basic_initialization(self):
        """Test creating EnvCheckpointState with reward parameters."""
        config = GameFactory.medium()
        state = EnvCheckpointState(
            completion_reward=1000.0,
            partial_completion_base=500.0,
            invalid_move_penalty=-10.0,
            singles_reduction_weight=5.0,
            game_config=config,
        )

        assert state.completion_reward == 1000.0
        assert state.partial_completion_base == 500.0
        assert state.invalid_move_penalty == -10.0
        assert state.singles_reduction_weight == 5.0
        assert state.game_config == config

    def test_game_config_integration(self):
        """Test that GameConfig is properly stored."""
        small_config = GameFactory.small()
        state = EnvCheckpointState(
            completion_reward=1000.0,
            partial_completion_base=500.0,
            invalid_move_penalty=-10.0,
            singles_reduction_weight=5.0,
            game_config=small_config,
        )

        assert state.game_config.num_rows == 5
        assert state.game_config.num_cols == 5
        assert state.game_config.num_colors == 3


class TestTrainingState:
    """Tests for TrainingState dataclass."""

    def test_basic_initialization(self):
        """Test creating TrainingState with training metadata."""
        state = TrainingState(
            total_epochs=1000,
            current_epoch=500,
            total_steps=50000,
            random_seed=42,
        )

        assert state.total_epochs == 1000
        assert state.current_epoch == 500
        assert state.total_steps == 50000
        assert state.random_seed == 42

    def test_optional_fields(self):
        """Test TrainingState with optional fields."""
        state = TrainingState(
            total_epochs=1000,
            current_epoch=500,
            total_steps=50000,
            random_seed=42,
            best_score=850.0,
            training_time_seconds=3600.0,
        )

        assert state.best_score == 850.0
        assert state.training_time_seconds == 3600.0


class TestCheckpointData:
    """Tests for CheckpointData dataclass."""

    def test_basic_initialization(self):
        """Test creating CheckpointData with required fields."""
        agent_state = AgentCheckpointState(
            epsilon=0.5,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            learning_rate=0.001,
            gamma=0.99,
            tau=0.005,
            batch_size=64,
            replay_buffer_size=10000,
        )

        env_state = EnvCheckpointState(
            completion_reward=1000.0,
            partial_completion_base=500.0,
            invalid_move_penalty=-10.0,
            singles_reduction_weight=5.0,
            game_config=GameFactory.medium(),
        )

        training_state = TrainingState(
            total_epochs=1000,
            current_epoch=500,
            total_steps=50000,
            random_seed=42,
        )

        timestamp = datetime.now()
        checkpoint = CheckpointData(
            experiment_name="test_experiment",
            epoch=500,
            timestamp=timestamp,
            model_weights_filename="model_epoch_500.pth",
            agent_state=agent_state,
            env_state=env_state,
            training_state=training_state,
            loss_history=[0.5, 0.4, 0.3, 0.2],
        )

        assert checkpoint.version == 1  # Default version
        assert checkpoint.experiment_name == "test_experiment"
        assert checkpoint.epoch == 500
        assert checkpoint.timestamp == timestamp
        assert checkpoint.model_weights_filename == "model_epoch_500.pth"
        assert checkpoint.agent_state == agent_state
        assert checkpoint.env_state == env_state
        assert checkpoint.training_state == training_state
        assert checkpoint.loss_history == [0.5, 0.4, 0.3, 0.2]
        assert checkpoint.benchmark_results is None
        assert checkpoint.replay_buffer_filename is None
        assert checkpoint.metadata == {}

    def test_with_optional_fields(self):
        """Test CheckpointData with optional benchmark and replay buffer."""
        agent_state = AgentCheckpointState(
            epsilon=0.3,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            learning_rate=0.001,
            gamma=0.99,
            tau=0.005,
            batch_size=64,
            replay_buffer_size=10000,
        )

        env_state = EnvCheckpointState(
            completion_reward=1000.0,
            partial_completion_base=500.0,
            invalid_move_penalty=-10.0,
            singles_reduction_weight=5.0,
            game_config=GameFactory.medium(),
        )

        training_state = TrainingState(
            total_epochs=1000,
            current_epoch=800,
            total_steps=80000,
            random_seed=42,
        )

        benchmark_results = {
            "mean_score": 750.0,
            "std_score": 50.0,
            "num_games": 100,
        }

        checkpoint = CheckpointData(
            experiment_name="test_experiment",
            epoch=800,
            timestamp=datetime.now(),
            model_weights_filename="model_epoch_800.pth",
            agent_state=agent_state,
            env_state=env_state,
            training_state=training_state,
            loss_history=[0.2, 0.15, 0.1],
            benchmark_results=benchmark_results,
            replay_buffer_filename="replay_buffer_epoch_800.pkl",
            metadata={"notes": "Good checkpoint", "experiment_id": "exp_001"},
        )

        assert checkpoint.benchmark_results == benchmark_results
        assert checkpoint.replay_buffer_filename == "replay_buffer_epoch_800.pkl"
        assert checkpoint.metadata["notes"] == "Good checkpoint"
        assert checkpoint.metadata["experiment_id"] == "exp_001"

    def test_version_field_for_format_evolution(self):
        """Test that version field supports checkpoint format evolution."""
        agent_state = AgentCheckpointState(
            epsilon=0.5,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            learning_rate=0.001,
            gamma=0.99,
            tau=0.005,
            batch_size=64,
            replay_buffer_size=10000,
        )

        env_state = EnvCheckpointState(
            completion_reward=1000.0,
            partial_completion_base=500.0,
            invalid_move_penalty=-10.0,
            singles_reduction_weight=5.0,
            game_config=GameFactory.medium(),
        )

        training_state = TrainingState(
            total_epochs=1000,
            current_epoch=500,
            total_steps=50000,
            random_seed=42,
        )

        # Default version should be 1
        checkpoint_v1 = CheckpointData(
            experiment_name="test",
            epoch=100,
            timestamp=datetime.now(),
            model_weights_filename="model.pth",
            agent_state=agent_state,
            env_state=env_state,
            training_state=training_state,
            loss_history=[],
        )
        assert checkpoint_v1.version == 1

        # Should be able to create with explicit version
        checkpoint_v2 = CheckpointData(
            version=2,
            experiment_name="test",
            epoch=100,
            timestamp=datetime.now(),
            model_weights_filename="model.pth",
            agent_state=agent_state,
            env_state=env_state,
            training_state=training_state,
            loss_history=[],
        )
        assert checkpoint_v2.version == 2

    def test_checkpoint_identifier(self):
        """Test generating unique checkpoint identifier."""
        checkpoint = CheckpointData(
            experiment_name="conv_model_experiment",
            epoch=500,
            timestamp=datetime.now(),
            model_weights_filename="model_epoch_500.pth",
            agent_state=AgentCheckpointState(
                epsilon=0.5,
                epsilon_min=0.01,
                epsilon_decay=0.995,
                learning_rate=0.001,
                gamma=0.99,
                tau=0.005,
                batch_size=64,
                replay_buffer_size=10000,
            ),
            env_state=EnvCheckpointState(
                completion_reward=1000.0,
                partial_completion_base=500.0,
                invalid_move_penalty=-10.0,
                singles_reduction_weight=5.0,
                game_config=GameFactory.medium(),
            ),
            training_state=TrainingState(
                total_epochs=1000,
                current_epoch=500,
                total_steps=50000,
                random_seed=42,
            ),
            loss_history=[],
        )

        # Should generate identifier like "conv_model_experiment_epoch_500"
        identifier = checkpoint.get_identifier()
        assert identifier == "conv_model_experiment_epoch_500"


class TestCheckpointSerialization:
    """Tests for checkpoint serialization and deserialization."""

    def test_checkpoint_pickle_round_trip(self):
        """Test that checkpoint can be pickled and unpickled."""
        import pickle

        agent_state = AgentCheckpointState(
            epsilon=0.42,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            learning_rate=0.001,
            gamma=0.99,
            tau=0.005,
            batch_size=64,
            replay_buffer_size=10000,
        )

        env_state = EnvCheckpointState(
            completion_reward=1000.0,
            partial_completion_base=500.0,
            invalid_move_penalty=-10.0,
            singles_reduction_weight=5.0,
            game_config=GameFactory.medium(),
        )

        training_state = TrainingState(
            total_epochs=1000,
            current_epoch=500,
            total_steps=50000,
            random_seed=42,
            best_score=850.0,
        )

        original_checkpoint = CheckpointData(
            version=1,
            experiment_name="pickle_test",
            epoch=500,
            timestamp=datetime(2025, 1, 15, 10, 30, 0),
            model_weights_filename="model_epoch_500.pth",
            agent_state=agent_state,
            env_state=env_state,
            training_state=training_state,
            loss_history=[0.5, 0.4, 0.3, 0.2],
            benchmark_results={"mean_score": 750.0},
            replay_buffer_filename="replay_buffer.pkl",
            metadata={"gpu": "A100"},
        )

        # Serialize to pickle
        pickled = pickle.dumps(original_checkpoint)

        # Deserialize from pickle
        restored_checkpoint = pickle.loads(pickled)

        # Verify all fields match
        assert restored_checkpoint.version == original_checkpoint.version
        assert restored_checkpoint.experiment_name == original_checkpoint.experiment_name
        assert restored_checkpoint.epoch == original_checkpoint.epoch
        assert restored_checkpoint.timestamp == original_checkpoint.timestamp
        assert restored_checkpoint.model_weights_filename == original_checkpoint.model_weights_filename
        assert restored_checkpoint.loss_history == original_checkpoint.loss_history
        assert restored_checkpoint.benchmark_results == original_checkpoint.benchmark_results
        assert restored_checkpoint.replay_buffer_filename == original_checkpoint.replay_buffer_filename
        assert restored_checkpoint.metadata == original_checkpoint.metadata

        # Verify agent state
        assert restored_checkpoint.agent_state.epsilon == 0.42
        assert restored_checkpoint.agent_state.gamma == 0.99
        assert restored_checkpoint.agent_state.batch_size == 64

        # Verify env state
        assert restored_checkpoint.env_state.completion_reward == 1000.0
        assert restored_checkpoint.env_state.game_config.num_rows == 8
        assert restored_checkpoint.env_state.game_config.num_cols == 8

        # Verify training state
        assert restored_checkpoint.training_state.current_epoch == 500
        assert restored_checkpoint.training_state.best_score == 850.0

    def test_game_config_serialization(self):
        """Test that GameConfig within checkpoint is properly serialized."""
        import pickle

        large_config = GameFactory.large()
        env_state = EnvCheckpointState(
            completion_reward=2000.0,
            partial_completion_base=1000.0,
            invalid_move_penalty=-20.0,
            singles_reduction_weight=10.0,
            game_config=large_config,
        )

        checkpoint = CheckpointData(
            experiment_name="config_test",
            epoch=100,
            timestamp=datetime.now(),
            model_weights_filename="model.pth",
            agent_state=AgentCheckpointState(
                epsilon=0.5,
                epsilon_min=0.01,
                epsilon_decay=0.995,
                learning_rate=0.001,
                gamma=0.99,
                tau=0.005,
                batch_size=64,
                replay_buffer_size=10000,
            ),
            env_state=env_state,
            training_state=TrainingState(
                total_epochs=1000,
                current_epoch=100,
                total_steps=10000,
                random_seed=42,
            ),
            loss_history=[],
        )

        # Serialize and deserialize
        restored = pickle.loads(pickle.dumps(checkpoint))

        # Verify game config dimensions
        assert restored.env_state.game_config.num_rows == 15
        assert restored.env_state.game_config.num_cols == 15
        assert restored.env_state.game_config.num_colors == 6

    def test_empty_optional_fields_serialization(self):
        """Test serialization with minimal required fields only."""
        import pickle

        checkpoint = CheckpointData(
            experiment_name="minimal_test",
            epoch=1,
            timestamp=datetime.now(),
            model_weights_filename="model_epoch_1.pth",
            agent_state=AgentCheckpointState(
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.995,
                learning_rate=0.001,
                gamma=0.99,
                tau=0.005,
                batch_size=64,
                replay_buffer_size=10000,
            ),
            env_state=EnvCheckpointState(
                completion_reward=1000.0,
                partial_completion_base=500.0,
                invalid_move_penalty=-10.0,
                singles_reduction_weight=5.0,
                game_config=GameFactory.small(),
            ),
            training_state=TrainingState(
                total_epochs=100,
                current_epoch=1,
                total_steps=1000,
                random_seed=42,
            ),
            loss_history=[],
        )

        # Serialize and deserialize
        restored = pickle.loads(pickle.dumps(checkpoint))

        # Verify optional fields are None/empty
        assert restored.benchmark_results is None
        assert restored.replay_buffer_filename is None
        assert restored.metadata == {}
        assert restored.loss_history == []
