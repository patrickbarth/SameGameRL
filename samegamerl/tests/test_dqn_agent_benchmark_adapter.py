"""Tests for DQN agent benchmark adapter."""

import numpy as np
import pytest
import torch
from torch import nn

from samegamerl.agents.dqn_agent import DqnAgent
from samegamerl.agents.dqn_agent_benchmark_adapter import DqnAgentBenchmarkAdapter
from samegamerl.game.game_config import GameFactory


class SimpleTestModel(nn.Module):
    """Minimal model for testing purposes."""

    def __init__(self, input_channels, num_rows, num_cols, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        input_size = input_channels * num_rows * num_cols
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)


class TestDqnAgentBenchmarkAdapter:
    """Test suite for DqnAgentBenchmarkAdapter."""

    @pytest.fixture
    def config(self):
        """Create small game config for testing."""
        return GameFactory.small()

    @pytest.fixture
    def agent(self, config):
        """Create simple DQN agent for testing."""
        model = SimpleTestModel(
            input_channels=config.num_colors,
            num_rows=config.num_rows,
            num_cols=config.num_cols,
            output_size=config.action_space_size
        )
        return DqnAgent(
            model=model,
            config=config,
            model_name="test_model",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1,
        )

    @pytest.fixture
    def adapter(self, agent):
        """Create adapter for testing."""
        return DqnAgentBenchmarkAdapter(agent)

    def test_adapter_initialization(self, adapter, agent):
        """Test adapter initializes correctly."""
        assert adapter.agent is agent
        assert adapter.config == agent.config
        assert hasattr(adapter, "name")
        assert isinstance(adapter.name, str)

    def test_epsilon_set_to_zero(self, adapter):
        """Test adapter forces epsilon to 0 for deterministic evaluation."""
        assert adapter.agent.epsilon == 0.0

    def test_model_in_eval_mode(self, adapter):
        """Test adapter puts model in evaluation mode."""
        assert not adapter.agent.model.training

    def test_name_format(self, adapter):
        """Test name follows expected format: modelname_timestamp_hash."""
        parts = adapter.name.split("_")
        assert len(parts) >= 3
        assert parts[0] == "test"
        assert parts[1] == "model"
        # Parts 2 and 3 are timestamp (YYYYMMDD_HHMMSS)
        # Part 4 is hash (8 chars)
        assert len(parts[-1]) == 8
        assert all(c in "0123456789abcdef" for c in parts[-1])

    def test_weight_hash_determinism(self, config):
        """Test same weights produce same hash."""
        model = SimpleTestModel(
            config.num_colors, config.num_rows, config.num_cols,
            config.action_space_size
        )

        hash1 = DqnAgentBenchmarkAdapter.compute_weight_hash(model)
        hash2 = DqnAgentBenchmarkAdapter.compute_weight_hash(model)

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 produces 32 hex chars

    def test_weight_hash_different_for_different_weights(self, config):
        """Test different weights produce different hashes."""
        model1 = SimpleTestModel(
            config.num_colors, config.num_rows, config.num_cols,
            config.action_space_size
        )
        model2 = SimpleTestModel(
            config.num_colors, config.num_rows, config.num_cols,
            config.action_space_size
        )

        # Models initialized with different random weights
        hash1 = DqnAgentBenchmarkAdapter.compute_weight_hash(model1)
        hash2 = DqnAgentBenchmarkAdapter.compute_weight_hash(model2)

        assert hash1 != hash2

    def test_board_to_observation_conversion(self, adapter, config):
        """Test board converts to correct one-hot observation."""
        board = [
            [1, 1, 0, 0, 0],
            [2, 1, 0, 0, 0],
            [2, 2, 0, 0, 0],
            [1, 2, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ]

        observation = adapter._board_to_observation(board)

        assert observation.shape == (config.num_colors, config.num_rows, config.num_cols)
        assert observation.dtype == np.float32

        # Check one-hot encoding for color 1
        assert observation[1, 0, 0] == 1  # Position (0,0) has color 1
        assert observation[1, 0, 1] == 1  # Position (0,1) has color 1
        assert observation[1, 1, 1] == 1  # Position (1,1) has color 1

        # Check one-hot encoding for color 2
        assert observation[2, 1, 0] == 1  # Position (1,0) has color 2
        assert observation[2, 2, 0] == 1  # Position (2,0) has color 2

        # Check empty cells (color 0)
        assert observation[0, 0, 2] == 1  # Position (0,2) is empty

    def test_select_action_returns_valid_tuple(self, adapter):
        """Test select_action returns (row, col) tuple."""
        board = [
            [1, 1, 0, 0, 0],
            [2, 1, 0, 0, 0],
            [2, 2, 0, 0, 0],
            [1, 2, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ]

        action = adapter.select_action(board)

        assert action is not None
        assert isinstance(action, tuple)
        assert len(action) == 2
        row, col = action
        assert 0 <= row < 5
        assert 0 <= col < 5

    def test_select_action_handles_any_agent_output(self, adapter, config):
        """Test select_action converts any agent action to coordinates."""
        board = [[0 for _ in range(config.num_cols)] for _ in range(config.num_rows)]
        board[0][0] = 1  # Only one tile

        # Mock agent to select a specific position
        original_act = adapter.agent.act

        def mock_act(obs):
            # Return action for position (1, 1)
            return config.num_cols + 1

        adapter.agent.act = mock_act

        action = adapter.select_action(board)

        # Adapter should return coordinates even for invalid tiles
        # (game.move() will handle validation)
        assert action == (1, 1)

        # Restore original method
        adapter.agent.act = original_act

    def test_action_1d_to_2d_conversion(self, adapter, config):
        """Test 1D action correctly converts to 2D coordinates."""
        # Test a few known conversions for 5x5 board (num_cols=5)
        test_cases = [
            (0, (0, 0)),    # First cell
            (4, (0, 4)),    # End of first row
            (5, (1, 0)),    # Start of second row
            (12, (2, 2)),   # Middle
            (24, (4, 4)),   # Last cell
        ]

        for action_1d, expected_2d in test_cases:
            row, col = divmod(action_1d, config.num_cols)
            assert (row, col) == expected_2d

    def test_deterministic_action_selection(self, adapter):
        """Test adapter produces same action for same board state."""
        board = [
            [1, 1, 0, 0, 0],
            [2, 1, 0, 0, 0],
            [2, 2, 0, 0, 0],
            [1, 2, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ]

        action1 = adapter.select_action(board)
        action2 = adapter.select_action(board)

        assert action1 == action2


class TestAdapterIntegration:
    """Integration tests with benchmark system components."""

    def test_adapter_compatible_with_benchmark_bot_base(self):
        """Test adapter implements BenchmarkBotBase interface."""
        config = GameFactory.small()
        model = SimpleTestModel(
            config.num_colors, config.num_rows, config.num_cols,
            config.action_space_size
        )
        agent = DqnAgent(
            model=model,
            config=config,
            model_name="test",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1,
        )
        adapter = DqnAgentBenchmarkAdapter(agent)

        # Check it has required attributes and methods
        assert hasattr(adapter, "name")
        assert hasattr(adapter, "select_action")
        assert callable(adapter.select_action)

    def test_multiple_adapters_have_unique_names(self):
        """Test creating multiple adapters produces unique names."""
        config = GameFactory.small()
        model1 = SimpleTestModel(
            config.num_colors, config.num_rows, config.num_cols,
            config.action_space_size
        )
        agent1 = DqnAgent(
            model=model1,
            config=config,
            model_name="test",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1,
        )

        model2 = SimpleTestModel(
            config.num_colors, config.num_rows, config.num_cols,
            config.action_space_size
        )
        agent2 = DqnAgent(
            model=model2,
            config=config,
            model_name="test",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1,
        )

        adapter1 = DqnAgentBenchmarkAdapter(agent1)
        adapter2 = DqnAgentBenchmarkAdapter(agent2)

        # Names should differ due to timestamp or hash
        assert adapter1.name != adapter2.name
