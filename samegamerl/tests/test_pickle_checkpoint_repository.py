"""Tests for pickle-based checkpoint repository."""

import tempfile
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
from samegamerl.training.pickle_checkpoint_repository import PickleCheckpointRepository


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoint tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_checkpoint():
    """Create a sample checkpoint for testing."""
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
        best_score=850.0,
    )

    return CheckpointData(
        version=1,
        experiment_name="test_experiment",
        epoch=500,
        timestamp=datetime(2025, 1, 15, 10, 30, 0),
        model_weights_filename="model_epoch_500.pth",
        agent_state=agent_state,
        env_state=env_state,
        training_state=training_state,
        loss_history=[0.5, 0.4, 0.3, 0.2],
        benchmark_results={"mean_score": 750.0},
        metadata={"notes": "Test checkpoint"},
    )


class TestPickleCheckpointRepository:
    """Tests for pickle checkpoint repository."""

    def test_save_checkpoint(self, temp_checkpoint_dir, sample_checkpoint):
        """Test saving a checkpoint to pickle file."""
        repo = PickleCheckpointRepository(temp_checkpoint_dir)
        checkpoint_id = repo.save(sample_checkpoint)

        # Verify checkpoint ID format
        assert checkpoint_id == "test_experiment_epoch_500"

        # Verify file was created
        checkpoint_file = temp_checkpoint_dir / f"{checkpoint_id}.pkl"
        assert checkpoint_file.exists()

    def test_load_checkpoint(self, temp_checkpoint_dir, sample_checkpoint):
        """Test loading a checkpoint from pickle file."""
        repo = PickleCheckpointRepository(temp_checkpoint_dir)
        checkpoint_id = repo.save(sample_checkpoint)

        # Load the checkpoint
        loaded = repo.load(checkpoint_id)

        # Verify all fields match
        assert loaded.version == sample_checkpoint.version
        assert loaded.experiment_name == sample_checkpoint.experiment_name
        assert loaded.epoch == sample_checkpoint.epoch
        assert loaded.timestamp == sample_checkpoint.timestamp
        assert loaded.model_weights_filename == sample_checkpoint.model_weights_filename
        assert loaded.loss_history == sample_checkpoint.loss_history
        assert loaded.benchmark_results == sample_checkpoint.benchmark_results
        assert loaded.metadata == sample_checkpoint.metadata

        # Verify agent state
        assert loaded.agent_state.epsilon == 0.5
        assert loaded.agent_state.gamma == 0.99
        assert loaded.agent_state.batch_size == 64

        # Verify env state
        assert loaded.env_state.completion_reward == 1000.0
        assert loaded.env_state.game_config.num_rows == 8

        # Verify training state
        assert loaded.training_state.current_epoch == 500
        assert loaded.training_state.best_score == 850.0

    def test_load_nonexistent_checkpoint(self, temp_checkpoint_dir):
        """Test loading a checkpoint that doesn't exist raises error."""
        repo = PickleCheckpointRepository(temp_checkpoint_dir)

        with pytest.raises(FileNotFoundError):
            repo.load("nonexistent_checkpoint")

    def test_list_checkpoints(self, temp_checkpoint_dir, sample_checkpoint):
        """Test listing all checkpoints in repository."""
        repo = PickleCheckpointRepository(temp_checkpoint_dir)

        # Initially empty
        assert repo.list_checkpoints() == []

        # Save multiple checkpoints
        checkpoint_1 = sample_checkpoint
        checkpoint_1.epoch = 100
        id_1 = repo.save(checkpoint_1)

        checkpoint_2 = sample_checkpoint
        checkpoint_2.epoch = 200
        id_2 = repo.save(checkpoint_2)

        checkpoint_3 = sample_checkpoint
        checkpoint_3.epoch = 300
        id_3 = repo.save(checkpoint_3)

        # List should return all IDs
        checkpoint_ids = repo.list_checkpoints()
        assert len(checkpoint_ids) == 3
        assert id_1 in checkpoint_ids
        assert id_2 in checkpoint_ids
        assert id_3 in checkpoint_ids

    def test_list_checkpoints_for_experiment(self, temp_checkpoint_dir, sample_checkpoint):
        """Test listing checkpoints filtered by experiment name."""
        repo = PickleCheckpointRepository(temp_checkpoint_dir)

        # Save checkpoints for different experiments
        checkpoint_exp1 = sample_checkpoint
        checkpoint_exp1.experiment_name = "experiment_1"
        checkpoint_exp1.epoch = 100
        id_exp1_100 = repo.save(checkpoint_exp1)

        checkpoint_exp1_200 = sample_checkpoint
        checkpoint_exp1_200.experiment_name = "experiment_1"
        checkpoint_exp1_200.epoch = 200
        id_exp1_200 = repo.save(checkpoint_exp1_200)

        checkpoint_exp2 = sample_checkpoint
        checkpoint_exp2.experiment_name = "experiment_2"
        checkpoint_exp2.epoch = 100
        id_exp2_100 = repo.save(checkpoint_exp2)

        # List all checkpoints for experiment_1
        exp1_checkpoints = repo.list_checkpoints(experiment_name="experiment_1")
        assert len(exp1_checkpoints) == 2
        assert id_exp1_100 in exp1_checkpoints
        assert id_exp1_200 in exp1_checkpoints
        assert id_exp2_100 not in exp1_checkpoints

    def test_delete_checkpoint(self, temp_checkpoint_dir, sample_checkpoint):
        """Test deleting a checkpoint."""
        repo = PickleCheckpointRepository(temp_checkpoint_dir)
        checkpoint_id = repo.save(sample_checkpoint)

        # Verify checkpoint exists
        assert checkpoint_id in repo.list_checkpoints()

        # Delete checkpoint
        repo.delete(checkpoint_id)

        # Verify checkpoint is gone
        assert checkpoint_id not in repo.list_checkpoints()
        checkpoint_file = temp_checkpoint_dir / f"{checkpoint_id}.pkl"
        assert not checkpoint_file.exists()

    def test_delete_nonexistent_checkpoint(self, temp_checkpoint_dir):
        """Test deleting nonexistent checkpoint raises error."""
        repo = PickleCheckpointRepository(temp_checkpoint_dir)

        with pytest.raises(FileNotFoundError):
            repo.delete("nonexistent_checkpoint")

    def test_checkpoint_exists(self, temp_checkpoint_dir, sample_checkpoint):
        """Test checking if checkpoint exists."""
        repo = PickleCheckpointRepository(temp_checkpoint_dir)

        # Initially doesn't exist
        assert not repo.exists("test_experiment_epoch_500")

        # Save checkpoint
        checkpoint_id = repo.save(sample_checkpoint)

        # Now exists
        assert repo.exists(checkpoint_id)

        # Delete and verify doesn't exist
        repo.delete(checkpoint_id)
        assert not repo.exists(checkpoint_id)

    def test_save_overwrites_existing_checkpoint(self, temp_checkpoint_dir, sample_checkpoint):
        """Test that saving with same ID overwrites existing checkpoint."""
        repo = PickleCheckpointRepository(temp_checkpoint_dir)

        # Save initial checkpoint
        checkpoint_id = repo.save(sample_checkpoint)
        initial_loaded = repo.load(checkpoint_id)
        assert initial_loaded.metadata["notes"] == "Test checkpoint"

        # Modify and save again with same ID
        sample_checkpoint.metadata["notes"] = "Updated checkpoint"
        repo.save(sample_checkpoint)

        # Load and verify it was updated
        updated_loaded = repo.load(checkpoint_id)
        assert updated_loaded.metadata["notes"] == "Updated checkpoint"

    def test_checkpoint_directory_creation(self, temp_checkpoint_dir):
        """Test that repository creates checkpoint directory if it doesn't exist."""
        nested_dir = temp_checkpoint_dir / "nested" / "checkpoints"
        assert not nested_dir.exists()

        # Create repository with non-existent directory
        repo = PickleCheckpointRepository(nested_dir)

        # Directory should be created
        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_save_checkpoint_with_special_characters(self, temp_checkpoint_dir, sample_checkpoint):
        """Test saving checkpoint with experiment name containing underscores."""
        sample_checkpoint.experiment_name = "conv_model_experiment"
        sample_checkpoint.epoch = 500

        repo = PickleCheckpointRepository(temp_checkpoint_dir)
        checkpoint_id = repo.save(sample_checkpoint)

        # Should handle underscores correctly
        assert checkpoint_id == "conv_model_experiment_epoch_500"
        assert repo.exists(checkpoint_id)

        # Should be loadable
        loaded = repo.load(checkpoint_id)
        assert loaded.experiment_name == "conv_model_experiment"
        assert loaded.epoch == 500
