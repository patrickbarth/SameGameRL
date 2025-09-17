import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from samegamerl.agents.base_agent import BaseAgent
from samegamerl.game.game_config import GameConfig
from samegamerl.agents.replay_buffer import ReplayBuffer


class DqnAgent(BaseAgent):
    """Deep Q-Network agent implementing DQN with experience replay and target networks.

    Supports epsilon-greedy exploration with decay scheduling and configurable
    hyperparameters for systematic experimentation across game variants.
    """

    def __init__(
        self,
        model: nn.Module,
        config: GameConfig,
        model_name: str,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        batch_size: int = 128,
        gamma: float = 0.95,
        tau: float = 0.005,
        replay_buffer_size: int = 50_000,
    ):
        # Store configuration
        self.config = config
        self.input_shape = config.observation_shape
        self.action_space_size = config.action_space_size

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using {self.device} device")

        self.model = model.to(self.device)
        self.target_model = deepcopy(model).to(self.device)

        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
        self.batch_size = batch_size
        self.won = 0

        # exploration
        self.epsilon = initial_epsilon  # Exploration rate
        self.epsilon_min = final_epsilon  # Minimal exploration rate
        self.epsilon_decay = epsilon_decay

        # learning
        self.gamma = gamma
        self.model_name = model_name
        self.tau = tau
        self.learning_rate = learning_rate

        self.criterion = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Path management
        self.models_dir = Path("samegamerl/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _create_tensor(self, data, dtype=torch.float32, requires_grad=False):
        """Create tensor directly on target device for efficiency."""
        return torch.tensor(
            data, dtype=dtype, device=self.device, requires_grad=requires_grad
        )

    def act(self, observation: np.ndarray) -> int:
        """Select action using epsilon-greedy policy with Q-value estimation."""
        was_training = self.model.training
        self.model.eval()

        if random.random() < self.epsilon:
            move = random.randint(0, self.action_space_size - 1)
        else:
            obs_tensor = self._create_tensor(observation).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(obs_tensor)
            move = q_values.argmax().item()

        self.model.train(was_training)
        return move

    def act_visualize(self, observation: np.ndarray) -> tuple[int, np.ndarray]:
        """Select action and return Q-values for visualization purposes."""
        was_training = self.model.training
        self.model.eval()

        obs_tensor = self._create_tensor(observation).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(obs_tensor)

        if random.random() < self.epsilon:
            move = random.randint(0, self.action_space_size - 1)
        else:
            move = q_values.argmax().item()

        self.model.train(was_training)
        return move, q_values

    def remember(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def learn(self) -> float:
        """Perform one gradient step on a batch of experiences from replay buffer."""

        if len(self.replay_buffer) < self.batch_size:
            return 0

        states = self.replay_buffer.sample(self.batch_size)
        obs, actions, rewards, next_obs, dones = states

        obs = self._create_tensor(obs)
        next_obs = self._create_tensor(next_obs)
        actions = self._create_tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = self._create_tensor(rewards).unsqueeze(1)
        dones = self._create_tensor(dones, dtype=torch.bool).unsqueeze(1)

        # Single gradient step with gradient clipping for stability
        pred_q_values = self.model(obs).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_model(next_obs).max(1).values.unsqueeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (~dones)

        loss = self.criterion(pred_q_values, target_q_values)
        self.opt.zero_grad()
        loss.backward()

        # Add gradient clipping to prevent gradient explosions
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.opt.step()
        return loss

    def decrease_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def update_target_model(self):
        """Soft update target network using polyak averaging with tau parameter."""
        target_model_state_dict = self.target_model.state_dict()
        model_state_dict = self.model.state_dict()
        for key in model_state_dict:
            target_model_state_dict[key] = model_state_dict[
                key
            ] * self.tau + target_model_state_dict[key] * (1 - self.tau)
        self.target_model.load_state_dict(target_model_state_dict)

    def save(self, name: str | None = None):
        """Save model with proper error handling and path management."""
        if not name:
            name = self.model_name

        try:
            model_path = self.models_dir / f"{name}.pth"
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.opt.state_dict(),
                    "target_model_state_dict": self.target_model.state_dict(),
                },
                model_path,
            )
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")
            raise

    def load(self, load_target=False, name: str | None = None):
        """Load model with proper error handling and path management."""
        if not name:
            name = self.model_name

        try:
            model_path = self.models_dir / f"{name}.pth"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])

            if load_target:
                self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
            else:
                self.target_model.load_state_dict(checkpoint["model_state_dict"])

            self.model.train()
            self.target_model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
