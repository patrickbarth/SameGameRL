"""Adapter to integrate DQN agents with the benchmark system."""

import hashlib
from datetime import datetime

import numpy as np
import torch

from samegamerl.agents.benchmark_bot_base import BenchmarkBotBase
from samegamerl.agents.dqn_agent import DqnAgent


class DqnAgentBenchmarkAdapter(BenchmarkBotBase):
    """
    Adapts DqnAgent to BenchmarkBotBase interface for standardized evaluation.

    Automatically generates unique bot names using model name, timestamp, and
    weight hash for tracking different training checkpoints.
    """

    def __init__(self, agent: DqnAgent):
        """
        Create adapter for DQN agent with automatic versioning.

        Args:
            agent: Trained DQN agent to evaluate
        """
        self.agent = agent
        self.config = agent.config

        # Force deterministic evaluation mode
        self.agent.epsilon = 0.0
        self.agent.model.eval()

        # Generate unique name: modelname_timestamp_hash
        weight_hash = self.compute_weight_hash(agent.model)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = f"{agent.model_name}_{timestamp}_{weight_hash[:8]}"

    def select_action(self, board: list[list[int]]) -> tuple[int, int] | None:
        """
        Select action using DQN agent's learned policy.

        Args:
            board: 2D list representing game board

        Returns:
            (row, col) tuple for action, or None if no valid moves
        """
        # Convert board to one-hot encoded observation
        observation = self._board_to_observation(board)

        # Get action from agent (1D action index)
        action_1d = self.agent.act(observation)

        # Convert 1D action to 2D coordinates
        row, col = divmod(action_1d, self.config.num_cols)

        return (row, col)

    def _board_to_observation(self, board: list[list[int]]) -> np.ndarray:
        """Convert board representation to one-hot encoded observation."""
        board_np = np.array(board)
        observation = np.zeros(
            (self.config.num_colors, self.config.num_rows, self.config.num_cols),
            dtype=np.float32
        )
        for color in range(self.config.num_colors):
            observation[color] = board_np == color
        return observation

    @staticmethod
    def compute_weight_hash(model: torch.nn.Module) -> str:
        """
        Compute deterministic hash of model weights.

        Args:
            model: PyTorch model to hash

        Returns:
            32-character hex string hash of model weights
        """
        # Get state dict and sort keys for determinism
        state_dict = model.state_dict()
        sorted_keys = sorted(state_dict.keys())

        # Create hash from all weight tensors
        hasher = hashlib.md5()
        for key in sorted_keys:
            tensor = state_dict[key]
            # Convert to bytes for hashing
            hasher.update(tensor.cpu().numpy().tobytes())

        return hasher.hexdigest()
