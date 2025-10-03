"""Tests for checkpoint state extractor pattern."""

import torch

from samegamerl.agents.dqn_agent import DqnAgent
from samegamerl.environments.samegame_env import SameGameEnv
from samegamerl.game.game_config import GameFactory
from samegamerl.training.checkpoint_state_extractor import CheckpointStateExtractor


class TestCheckpointStateExtractor:
    """Tests for extracting checkpoint state from domain objects."""

    def test_extract_agent_state(self):
        """Test extracting agent hyperparameters and training state."""
        # Create a simple model for the agent
        config = GameFactory.small()

        class SimpleModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.fc = torch.nn.Linear(config.total_cells * config.num_colors, config.action_space_size)

            def forward(self, x):
                return self.fc(x.flatten(start_dim=1))

        model = SimpleModel(config)

        agent = DqnAgent(
            model=model,
            config=config,
            model_name="test_model",
            learning_rate=0.001,
            gamma=0.99,
            initial_epsilon=0.8,
            final_epsilon=0.01,
            epsilon_decay=0.995,
            tau=0.005,
            batch_size=32,
            replay_buffer_size=5000,
        )

        # Extract state
        extractor = CheckpointStateExtractor()
        agent_state = extractor.extract_agent_state(agent)

        # Verify extracted values
        assert agent_state.epsilon == 0.8
        assert agent_state.epsilon_min == 0.01
        assert agent_state.epsilon_decay == 0.995
        assert agent_state.learning_rate == 0.001
        assert agent_state.gamma == 0.99
        assert agent_state.tau == 0.005
        assert agent_state.batch_size == 32
        assert agent_state.replay_buffer_size == 5000

    def test_extract_agent_state_after_epsilon_decay(self):
        """Test extracting agent state captures current epsilon value."""
        config = GameFactory.small()

        class SimpleModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.fc = torch.nn.Linear(config.total_cells * config.num_colors, config.action_space_size)

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

        # Simulate epsilon decay
        agent.epsilon = 0.5  # Simulate after training

        extractor = CheckpointStateExtractor()
        agent_state = extractor.extract_agent_state(agent)

        # Should capture current decayed epsilon, not initial
        assert agent_state.epsilon == 0.5

    def test_extract_env_state(self):
        """Test extracting environment configuration and reward parameters."""
        config = GameFactory.medium()
        env = SameGameEnv(
            config=config,
            completion_reward=1000.0,
            partial_completion_base=500.0,
            invalid_move_penalty=-10.0,
            singles_reduction_weight=5.0,
        )

        extractor = CheckpointStateExtractor()
        env_state = extractor.extract_env_state(env)

        # Verify reward parameters
        assert env_state.completion_reward == 1000.0
        assert env_state.partial_completion_base == 500.0
        assert env_state.invalid_move_penalty == -10.0
        assert env_state.singles_reduction_weight == 5.0

        # Verify game config
        assert env_state.game_config == config
        assert env_state.game_config.num_rows == 8
        assert env_state.game_config.num_cols == 8
        assert env_state.game_config.num_colors == 4

    def test_extract_env_state_with_custom_config(self):
        """Test extracting env state with custom game configuration."""
        custom_config = GameFactory.custom(num_rows=10, num_cols=12, num_colors=5)
        env = SameGameEnv(
            config=custom_config,
            completion_reward=2000.0,
            partial_completion_base=1000.0,
            invalid_move_penalty=-20.0,
            singles_reduction_weight=10.0,
        )

        extractor = CheckpointStateExtractor()
        env_state = extractor.extract_env_state(env)

        assert env_state.game_config.num_rows == 10
        assert env_state.game_config.num_cols == 12
        assert env_state.game_config.num_colors == 5

    def test_restore_agent_state(self):
        """Test restoring agent state from checkpoint."""
        config = GameFactory.small()

        class SimpleModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.fc = torch.nn.Linear(config.total_cells * config.num_colors, config.action_space_size)

            def forward(self, x):
                return self.fc(x.flatten(start_dim=1))

        model = SimpleModel(config)

        # Create agent with initial values
        agent = DqnAgent(
            model=model,
            config=config,
            model_name="test_model",
            initial_epsilon=1.0,
            final_epsilon=0.01,
            epsilon_decay=0.995,
            learning_rate=0.001,
            gamma=0.99,
            tau=0.005,
        )

        # Extract state
        extractor = CheckpointStateExtractor()
        saved_state = extractor.extract_agent_state(agent)

        # Modify agent (simulate training)
        agent.epsilon = 0.3
        agent.gamma = 0.95

        # Restore from saved state
        extractor.restore_agent_state(saved_state, agent)

        # Verify restoration
        assert agent.epsilon == 1.0
        assert agent.gamma == 0.99
        assert agent.tau == 0.005

    def test_extract_and_restore_round_trip(self):
        """Test extracting and restoring agent state preserves all values."""
        config = GameFactory.medium()

        class SimpleModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.fc = torch.nn.Linear(config.total_cells * config.num_colors, config.action_space_size)

            def forward(self, x):
                return self.fc(x.flatten(start_dim=1))

        model = SimpleModel(config)

        original_agent = DqnAgent(
            model=model,
            config=config,
            model_name="test_model",
            initial_epsilon=0.42,
            final_epsilon=0.01,
            epsilon_decay=0.995,
            learning_rate=0.0005,
            gamma=0.98,
            tau=0.003,
            batch_size=128,
            replay_buffer_size=20000,
        )

        # Extract state
        extractor = CheckpointStateExtractor()
        state = extractor.extract_agent_state(original_agent)

        # Create new agent with different values
        new_agent = DqnAgent(
            model=SimpleModel(config),
            config=config,
            model_name="test_model",
            learning_rate=0.001,
            initial_epsilon=0.1,
            final_epsilon=0.01,
            epsilon_decay=0.995,
            gamma=0.5,
        )

        # Restore state to new agent
        extractor.restore_agent_state(state, new_agent)

        # Verify all values match original
        assert new_agent.epsilon == 0.42
        assert new_agent.epsilon_min == 0.01
        assert new_agent.epsilon_decay == 0.995
        assert new_agent.learning_rate == 0.0005
        assert new_agent.gamma == 0.98
        assert new_agent.tau == 0.003
        assert new_agent.batch_size == 128
