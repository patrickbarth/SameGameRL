"""Tests for checkpoint service."""

import tempfile
from pathlib import Path

import pytest
import torch

from samegamerl.agents.dqn_agent import DqnAgent
from samegamerl.environments.samegame_env import SameGameEnv
from samegamerl.game.game_config import GameFactory
from samegamerl.training.checkpoint_service import CheckpointService
from samegamerl.training.pickle_checkpoint_repository import PickleCheckpointRepository


@pytest.fixture
def temp_dirs():
    """Create temporary directories for checkpoints and models."""
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        with tempfile.TemporaryDirectory() as model_dir:
            yield Path(checkpoint_dir), Path(model_dir)


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
        initial_epsilon=0.8,
        final_epsilon=0.01,
        epsilon_decay=0.995,
    )

    env = SameGameEnv(
        config=config,
        completion_reward=1000.0,
        partial_completion_base=500.0,
        invalid_move_penalty=-10.0,
        singles_reduction_weight=5.0,
    )

    return agent, env


class TestCheckpointService:
    """Tests for checkpoint service."""

    def test_create_checkpoint(self, temp_dirs, agent_and_env):
        """Test creating a checkpoint."""
        checkpoint_dir, model_dir = temp_dirs
        agent, env = agent_and_env

        repository = PickleCheckpointRepository(checkpoint_dir)
        service = CheckpointService(repository, model_dir)

        checkpoint_id = service.create_checkpoint(
            agent=agent,
            env=env,
            experiment_name="test_experiment",
            epoch=100,
            total_epochs=1000,
            total_steps=10000,
            loss_history=[0.5, 0.4, 0.3],
            random_seed=42,
        )

        # Verify checkpoint ID format
        assert checkpoint_id == "test_experiment_epoch_100"

        # Verify checkpoint was saved
        assert repository.exists(checkpoint_id)

        # Verify model weights were saved
        model_file = model_dir / "test_experiment_epoch_100_model.pth"
        assert model_file.exists()

    def test_load_checkpoint(self, temp_dirs, agent_and_env):
        """Test loading a checkpoint."""
        checkpoint_dir, model_dir = temp_dirs
        agent, env = agent_and_env

        # Modify agent state to simulate training
        agent.epsilon = 0.5

        repository = PickleCheckpointRepository(checkpoint_dir)
        service = CheckpointService(repository, model_dir)

        # Create checkpoint
        checkpoint_id = service.create_checkpoint(
            agent=agent,
            env=env,
            experiment_name="test_experiment",
            epoch=100,
            total_epochs=1000,
            total_steps=10000,
            loss_history=[0.5, 0.4, 0.3],
            random_seed=42,
        )

        # Load checkpoint
        loaded_checkpoint = service.load_checkpoint(checkpoint_id)

        # Verify checkpoint data
        assert loaded_checkpoint.experiment_name == "test_experiment"
        assert loaded_checkpoint.epoch == 100
        assert loaded_checkpoint.agent_state.epsilon == 0.5
        assert loaded_checkpoint.training_state.total_epochs == 1000
        assert loaded_checkpoint.training_state.total_steps == 10000
        assert loaded_checkpoint.loss_history == [0.5, 0.4, 0.3]

    def test_create_checkpoint_with_benchmark_results(self, temp_dirs, agent_and_env):
        """Test creating checkpoint with benchmark results."""
        checkpoint_dir, model_dir = temp_dirs
        agent, env = agent_and_env

        repository = PickleCheckpointRepository(checkpoint_dir)
        service = CheckpointService(repository, model_dir)

        benchmark_results = {
            "mean_score": 750.0,
            "std_score": 50.0,
            "num_games": 100,
        }

        checkpoint_id = service.create_checkpoint(
            agent=agent,
            env=env,
            experiment_name="test_experiment",
            epoch=200,
            total_epochs=1000,
            total_steps=20000,
            loss_history=[0.3, 0.2, 0.1],
            benchmark_results=benchmark_results,
            random_seed=42,
        )

        # Load and verify benchmark results
        loaded = service.load_checkpoint(checkpoint_id)
        assert loaded.benchmark_results == benchmark_results

    def test_checkpoint_captures_current_agent_state(self, temp_dirs, agent_and_env):
        """Test that checkpoint captures agent state at checkpoint time."""
        checkpoint_dir, model_dir = temp_dirs
        agent, env = agent_and_env

        repository = PickleCheckpointRepository(checkpoint_dir)
        service = CheckpointService(repository, model_dir)

        # Simulate epsilon decay during training
        agent.epsilon = 0.6

        checkpoint_id = service.create_checkpoint(
            agent=agent,
            env=env,
            experiment_name="test_experiment",
            epoch=300,
            total_epochs=1000,
            total_steps=30000,
            loss_history=[],
            random_seed=42,
        )

        # Load checkpoint
        loaded = service.load_checkpoint(checkpoint_id)

        # Should capture decayed epsilon value
        assert loaded.agent_state.epsilon == 0.6

    def test_checkpoint_captures_env_state(self, temp_dirs, agent_and_env):
        """Test that checkpoint captures environment configuration."""
        checkpoint_dir, model_dir = temp_dirs
        agent, env = agent_and_env

        repository = PickleCheckpointRepository(checkpoint_dir)
        service = CheckpointService(repository, model_dir)

        checkpoint_id = service.create_checkpoint(
            agent=agent,
            env=env,
            experiment_name="test_experiment",
            epoch=100,
            total_epochs=1000,
            total_steps=10000,
            loss_history=[],
            random_seed=42,
        )

        loaded = service.load_checkpoint(checkpoint_id)

        # Verify environment state
        assert loaded.env_state.completion_reward == 1000.0
        assert loaded.env_state.singles_reduction_weight == 5.0
        assert loaded.env_state.game_config.num_rows == 5
        assert loaded.env_state.game_config.num_colors == 3

    def test_multiple_checkpoints_for_same_experiment(self, temp_dirs, agent_and_env):
        """Test creating multiple checkpoints for the same experiment."""
        checkpoint_dir, model_dir = temp_dirs
        agent, env = agent_and_env

        repository = PickleCheckpointRepository(checkpoint_dir)
        service = CheckpointService(repository, model_dir)

        # Create checkpoints at different epochs
        id_100 = service.create_checkpoint(
            agent, env, "experiment_1", 100, 1000, 10000, [], random_seed=42
        )
        id_200 = service.create_checkpoint(
            agent, env, "experiment_1", 200, 1000, 20000, [], random_seed=42
        )
        id_300 = service.create_checkpoint(
            agent, env, "experiment_1", 300, 1000, 30000, [], random_seed=42
        )

        # Verify all checkpoints exist
        assert repository.exists(id_100)
        assert repository.exists(id_200)
        assert repository.exists(id_300)

        # Verify they can be listed
        checkpoints = repository.list_checkpoints(experiment_name="experiment_1")
        assert len(checkpoints) == 3
