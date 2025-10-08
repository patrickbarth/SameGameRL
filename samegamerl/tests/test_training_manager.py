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

    def test_warmup_populates_buffer(self, agent_and_env):
        """Test that warmup populates the replay buffer."""
        agent, env = agent_and_env

        manager = TrainingManager(agent, env, experiment_name="test_exp")

        # Buffer should be empty initially
        initial_buffer_size = len(agent.replay_buffer)
        assert initial_buffer_size == 0

        # Warmup should fill buffer
        manager.warmup(episodes=10)

        # Buffer should now have experiences
        assert len(agent.replay_buffer) > initial_buffer_size

    def test_create_checkpoint_returns_id(self, temp_checkpoint_dir, agent_and_env):
        """Test that create_checkpoint returns valid checkpoint ID."""
        agent, env = agent_and_env

        repository = PickleCheckpointRepository(temp_checkpoint_dir)
        checkpoint_service = CheckpointService(repository)

        manager = TrainingManager(
            agent, env, experiment_name="test_exp", checkpoint_service=checkpoint_service
        )

        # Train a bit
        manager.train(epochs=5)

        # Create checkpoint
        checkpoint_id = manager.create_checkpoint()

        # Should return valid ID
        assert checkpoint_id == "test_exp_epoch_5"
        assert repository.exists(checkpoint_id)

    def test_rollback_restores_state(self, temp_checkpoint_dir, agent_and_env):
        """Test that rollback restores model, optimizer, and epsilon."""
        agent, env = agent_and_env

        repository = PickleCheckpointRepository(temp_checkpoint_dir)
        checkpoint_service = CheckpointService(repository)

        manager = TrainingManager(
            agent, env, experiment_name="test_exp", checkpoint_service=checkpoint_service
        )

        # Train and checkpoint
        manager.warmup(episodes=10)
        manager.train(epochs=5)
        checkpoint_id = manager.create_checkpoint()

        # Save state at checkpoint
        checkpoint_epsilon = agent.epsilon
        checkpoint_epoch = manager.current_epoch
        checkpoint_model_weights = agent.model.state_dict()["fc.weight"].clone()

        # Continue training (epsilon may decay, model changes)
        manager.train(epochs=10)

        # Verify state changed
        assert agent.epsilon <= checkpoint_epsilon  # May have hit min epsilon
        assert manager.current_epoch == 15
        assert not torch.equal(
            agent.model.state_dict()["fc.weight"], checkpoint_model_weights
        )

        # Rollback
        manager.rollback_to_checkpoint(checkpoint_id)

        # Verify state restored (epsilon goes back up if it decayed)
        assert agent.epsilon == checkpoint_epsilon
        assert manager.current_epoch == checkpoint_epoch
        assert torch.allclose(
            agent.model.state_dict()["fc.weight"],
            checkpoint_model_weights,
            atol=1e-6,
        )

    def test_rollback_clears_buffer(self, temp_checkpoint_dir, agent_and_env):
        """Test that rollback clears the replay buffer."""
        agent, env = agent_and_env

        repository = PickleCheckpointRepository(temp_checkpoint_dir)
        checkpoint_service = CheckpointService(repository)

        manager = TrainingManager(
            agent, env, experiment_name="test_exp", checkpoint_service=checkpoint_service
        )

        # Warmup and checkpoint
        manager.warmup(episodes=10)
        manager.train(epochs=5)
        checkpoint_id = manager.create_checkpoint()

        # Buffer should have experiences
        assert len(agent.replay_buffer) > 0

        # Rollback
        manager.rollback_to_checkpoint(checkpoint_id)

        # Buffer should be empty
        assert len(agent.replay_buffer) == 0

    def test_adaptive_training_workflow(self, temp_checkpoint_dir, agent_and_env):
        """Test complete adaptive training workflow: train → checkpoint → train → rollback → warmup → train."""
        agent, env = agent_and_env

        repository = PickleCheckpointRepository(temp_checkpoint_dir)
        checkpoint_service = CheckpointService(repository)

        manager = TrainingManager(
            agent, env, experiment_name="test_exp", checkpoint_service=checkpoint_service
        )

        # Initial training
        manager.warmup(episodes=5)
        manager.train(epochs=10)
        checkpoint_id = manager.create_checkpoint()

        assert manager.current_epoch == 10

        # Continue training
        manager.train(epochs=5)
        assert manager.current_epoch == 15

        # Rollback to earlier state
        manager.rollback_to_checkpoint(checkpoint_id)
        assert manager.current_epoch == 10
        assert len(agent.replay_buffer) == 0

        # Warmup again from rolled-back state
        manager.warmup(episodes=5)
        assert len(agent.replay_buffer) > 0

        # Continue training from rollback point
        manager.train(epochs=5)
        assert manager.current_epoch == 15

    def test_checkpoint_captures_training_progress(self, temp_checkpoint_dir, agent_and_env):
        """Test that checkpoints capture training progress correctly."""
        agent, env = agent_and_env

        repository = PickleCheckpointRepository(temp_checkpoint_dir)
        checkpoint_service = CheckpointService(repository)

        manager = TrainingManager(
            agent, env, experiment_name="test_exp", checkpoint_service=checkpoint_service
        )

        # Train and create checkpoint
        manager.train(epochs=10)
        checkpoint_id = manager.create_checkpoint()

        # Load checkpoint
        checkpoint = checkpoint_service.load_checkpoint(checkpoint_id)

        # Verify training state
        assert checkpoint.training_state.current_epoch == 10
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

