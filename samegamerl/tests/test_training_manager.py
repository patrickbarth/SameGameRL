"""Tests for training manager."""

import tempfile
from pathlib import Path

import pytest
import torch

from samegamerl.agents.dqn_agent import DqnAgent
from samegamerl.environments.samegame_env import SameGameEnv
from samegamerl.game.game_config import GameFactory
from samegamerl.training.checkpoint_service import CheckpointService
from samegamerl.training.pickle_checkpoint_repository import PickleCheckpointRepository
from samegamerl.training.training_manager import TrainingManager


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        yield Path(checkpoint_dir)


@pytest.fixture
def agent_and_env():
    """Create a simple agent and environment for testing."""
    config = GameFactory.small()

    class SimpleModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.fc = torch.nn.Linear(
                config.total_cells * config.num_colors, config.action_space_size
            )

        def forward(self, x):
            return self.fc(x.flatten(start_dim=1))

    model = SimpleModel(config)
    agent = DqnAgent(
        model=model,
        config=config,
        model_name="test_model",
        learning_rate=0.001,
        initial_epsilon=1.0,
        final_epsilon=0.01,
        epsilon_decay=0.995,
    )

    env = SameGameEnv(config=config)

    return agent, env


class TestTrainingManager:
    """Tests for training manager."""

    def test_basic_training(self, agent_and_env):
        """Test basic training without checkpointing."""
        agent, env = agent_and_env

        manager = TrainingManager(agent, env, experiment_name="test_exp")

        # Train for a few epochs without checkpointing
        loss_history = manager.train(epochs=5)

        # Should return loss history
        assert isinstance(loss_history, list)
        assert len(loss_history) > 0

    def test_training_with_checkpoints(self, temp_checkpoint_dir, agent_and_env):
        """Test training with periodic checkpointing."""
        agent, env = agent_and_env

        repository = PickleCheckpointRepository(temp_checkpoint_dir)
        checkpoint_service = CheckpointService(repository)

        manager = TrainingManager(
            agent, env, experiment_name="test_exp", checkpoint_service=checkpoint_service
        )

        # Train with checkpoints every 5 epochs
        loss_history = manager.train_with_checkpoints(
            total_epochs=15, checkpoint_every=5, random_seed=42
        )

        # Should have 3 checkpoints (epochs 5, 10, 15)
        checkpoints = repository.list_checkpoints(experiment_name="test_exp")
        assert len(checkpoints) == 3
        assert "test_exp_epoch_5" in checkpoints
        assert "test_exp_epoch_10" in checkpoints
        assert "test_exp_epoch_15" in checkpoints

        # Should return loss history
        assert len(loss_history) > 0

    def test_checkpoint_captures_training_progress(self, temp_checkpoint_dir, agent_and_env):
        """Test that checkpoints capture training progress correctly."""
        agent, env = agent_and_env

        repository = PickleCheckpointRepository(temp_checkpoint_dir)
        checkpoint_service = CheckpointService(repository)

        manager = TrainingManager(
            agent, env, experiment_name="test_exp", checkpoint_service=checkpoint_service
        )

        # Train with checkpoint at epoch 10
        manager.train_with_checkpoints(
            total_epochs=10, checkpoint_every=10, random_seed=42
        )

        # Load checkpoint
        checkpoint = checkpoint_service.load_checkpoint("test_exp_epoch_10")

        # Verify training state
        assert checkpoint.training_state.current_epoch == 10
        assert checkpoint.training_state.total_epochs == 10
        assert checkpoint.training_state.random_seed == 42

    def test_epsilon_decay_during_training(self, agent_and_env):
        """Test that epsilon decays during training."""
        agent, env = agent_and_env

        initial_epsilon = agent.epsilon

        manager = TrainingManager(agent, env, experiment_name="test_exp")

        # Train for some epochs
        manager.train(epochs=10)

        # Epsilon should have decayed
        assert agent.epsilon < initial_epsilon

    def test_multiple_checkpoint_intervals(self, temp_checkpoint_dir, agent_and_env):
        """Test training with different checkpoint intervals."""
        agent, env = agent_and_env

        repository = PickleCheckpointRepository(temp_checkpoint_dir)
        checkpoint_service = CheckpointService(repository)

        manager = TrainingManager(
            agent, env, experiment_name="test_exp", checkpoint_service=checkpoint_service
        )

        # Train with checkpoints every 3 epochs
        manager.train_with_checkpoints(
            total_epochs=9, checkpoint_every=3, random_seed=42
        )

        # Should have 3 checkpoints (epochs 3, 6, 9)
        checkpoints = repository.list_checkpoints(experiment_name="test_exp")
        assert len(checkpoints) == 3
        assert "test_exp_epoch_3" in checkpoints
        assert "test_exp_epoch_6" in checkpoints
        assert "test_exp_epoch_9" in checkpoints
