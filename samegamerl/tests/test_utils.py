"""
Additional test utilities and helpers for SameGameRL testing.

This module provides specialized testing utilities that complement conftest.py,
focusing on domain-specific testing helpers and validation functions.

This file contains:
- Builder patterns for creating complex test objects (GameTestBuilder, EnvironmentTestBuilder, AgentTestBuilder)
- Specialized validators for game mechanics (GameStateValidator, EnvironmentValidator, AgentValidator)
- Training and performance validation utilities
- Integration test helpers for complete episodes and training workflows
- Advanced mock objects with specific behaviors

For basic fixtures and simple utilities, see conftest.py which provides:
- Static board configurations and simple validation functions
- Basic mock objects and test models
- Standard pytest fixtures
"""

import gc
import numpy as np
import torch
import time
import psutil

from samegamerl.game.game import Game
from samegamerl.game.game_config import GameConfig, GameFactory
from samegamerl.environments.samegame_env import SameGameEnv
from samegamerl.agents.dqn_agent import DqnAgent

try:
    from .conftest import TEST_BOARD_CONFIGS
except ImportError:
    TEST_BOARD_CONFIGS = {}


class GameTestBuilder:
    """Builder pattern for creating test games with specific configurations"""
    
    def __init__(self):
        self.config = GameFactory.default()
        self.board = None
    
    def with_dimensions(self, rows: int, cols: int, colors: int):
        """Set game dimensions"""
        self.config = GameFactory.custom(rows, cols, colors)
        return self
    
    def with_board(self, board: list[list[int]]):
        """Set specific board configuration"""
        self.board = board
        return self
    
    def with_predefined_board(self, config_name: str):
        """Use a predefined board configuration from TEST_BOARD_CONFIGS"""
        if config_name in TEST_BOARD_CONFIGS:
            self.board = TEST_BOARD_CONFIGS[config_name]
        else:
            raise ValueError(f"Unknown board configuration: {config_name}")
        return self
    
    def with_empty_board(self):
        """Create an empty board"""
        self.board = [[0] * self.config.num_cols for _ in range(self.config.num_rows)]
        return self
    
    def with_single_color_board(self, color: int = 1):
        """Create a board with all tiles the same color"""
        self.board = [[color] * self.config.num_cols for _ in range(self.config.num_rows)]
        return self
    
    def with_checkerboard(self, color1: int = 1, color2: int = 2):
        """Create a checkerboard pattern"""
        if "checkerboard_2x2" in TEST_BOARD_CONFIGS and self.config.num_rows == 2 and self.config.num_cols == 2:
            self.board = TEST_BOARD_CONFIGS["checkerboard_2x2"]
        else:
            self.board = []
            for r in range(self.config.num_rows):
                row = []
                for c in range(self.config.num_cols):
                    color = color1 if (r + c) % 2 == 0 else color2
                    row.append(color)
                self.board.append(row)
        return self
    
    def build(self) -> Game:
        """Build the configured game"""
        game = Game(self.config)
        
        if self.board:
            game.set_board(self.board)
        
        return game


class EnvironmentTestBuilder:
    """Builder pattern for creating test environments"""
    
    def __init__(self):
        self.config = GameFactory.default()
        self.initial_board = None
    
    def with_dimensions(self, rows: int, cols: int, colors: int):
        self.config = GameFactory.custom(rows, cols, colors)
        return self
    
    def with_initial_board(self, board: list[list[int]]):
        self.initial_board = board
        return self
    
    def build(self) -> SameGameEnv:
        env = SameGameEnv(self.config)
        if self.initial_board:
            env.reset(board=self.initial_board)
        return env


class AgentTestBuilder:
    """Builder pattern for creating test agents"""
    
    def __init__(self, config: GameConfig | None = None):
        if config is None:
            config = GameFactory.default()
        self.config = config
        self.model = None
        self.model_name = "test_agent"
        self.learning_rate = 0.001
        self.initial_epsilon = 1.0
        self.epsilon_decay = 0.001
        self.final_epsilon = 0.1
        self.batch_size = 128
        self.gamma = 0.95
        self.tau = 0.5
    
    def with_model(self, model: torch.nn.Module):
        self.model = model
        return self
    
    def with_simple_model(self):
        """Use a simple linear model for testing"""
        try:
            from .conftest import TinyTestModel
            input_size = self.config.num_rows * self.config.num_cols * self.config.num_colors
            self.model = TinyTestModel(input_size, self.config.action_space_size)
        except ImportError:
            # Fallback if conftest not available
            import torch.nn as nn
            input_size = self.config.num_rows * self.config.num_cols * self.config.num_colors
            output_size = self.config.action_space_size
            
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(input_size, output_size)
                
                def forward(self, x):
                    return self.fc(torch.flatten(x, start_dim=1))
            
            self.model = SimpleModel()
        return self
    
    def with_hyperparameters(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def build(self) -> DqnAgent:
        if self.model is None:
            self.with_simple_model()
        
        assert self.model is not None
        
        return DqnAgent(
            model=self.model,
            config=self.config,
            model_name=self.model_name,
            learning_rate=self.learning_rate,
            initial_epsilon=self.initial_epsilon,
            epsilon_decay=self.epsilon_decay,
            final_epsilon=self.final_epsilon,
            batch_size=self.batch_size,
            gamma=self.gamma,
            tau=self.tau
        )


class GameStateValidator:
    """Utilities for validating game state consistency"""
    
    @staticmethod
    def validate_board_integrity(game: Game):
        """Validate that game board maintains integrity"""
        board = game.get_board()
        
        # Check dimensions
        assert len(board) == game.config.num_rows
        assert all(len(row) == game.config.num_cols for row in board)
        
        # Check color values
        for row in board:
            for cell in row:
                assert 0 <= cell < game.config.num_colors
        
        # Check left count matches actual non-empty cells
        actual_left = sum(1 for row in board for cell in row if cell != 0)
        assert game.left == actual_left, f"Left count {game.left} != actual {actual_left}"
    
    @staticmethod
    def validate_physics_applied(game: Game):
        """Validate that physics (sinking and shrinking) have been applied"""
        board = game.get_board()
        
        # Check sinking: no floating tiles
        for c in range(game.config.num_cols):
            found_empty = False
            for r in range(game.config.num_rows - 1, -1, -1):
                if board[r][c] == 0:
                    found_empty = True
                elif found_empty:
                    # Found non-empty cell below empty cell
                    assert False, f"Floating tile at ({r}, {c})"
        
        # Check shrinking: no empty columns between non-empty columns
        last_non_empty_col = -1
        for c in range(game.config.num_cols):
            col_has_tiles = any(board[r][c] != 0 for r in range(game.config.num_rows))
            if col_has_tiles:
                # Check if there are empty columns before this one after last_non_empty_col
                for empty_c in range(last_non_empty_col + 1, c):
                    col_empty = all(board[r][empty_c] == 0 for r in range(game.config.num_rows))
                    assert col_empty, f"Non-empty column {c} found after empty columns"
                last_non_empty_col = c
    
    @staticmethod
    def validate_connected_component(game: Game, start_pos: tuple[int, int]) -> list[tuple[int, int]]:
        """Find and validate connected component from starting position"""
        board = game.get_board()
        if start_pos[0] < 0 or start_pos[0] >= game.config.num_rows or \
           start_pos[1] < 0 or start_pos[1] >= game.config.num_cols:
            return []
        
        start_color = board[start_pos[0]][start_pos[1]]
        if start_color == 0:
            return []
        
        visited = set()
        component = []
        stack = [start_pos]
        
        while stack:
            pos = stack.pop()
            if pos in visited:
                continue
            
            row, col = pos
            if (0 <= row < game.config.num_rows and 0 <= col < game.config.num_cols and
                board[row][col] == start_color):
                visited.add(pos)
                component.append(pos)
                
                # Add neighbors
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = (row + dr, col + dc)
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return component


class EnvironmentValidator:
    """Utilities for validating environment behavior"""
    
    @staticmethod
    def validate_observation_shape(env: SameGameEnv, obs: np.ndarray):
        """Validate observation has correct shape and properties"""
        from .conftest import assert_valid_observation
        expected_shape = (env.config.num_colors, env.config.num_rows, env.config.num_cols)
        assert_valid_observation(obs, expected_shape)
    
    @staticmethod
    def validate_step_contract(env: SameGameEnv, step_result: tuple[np.ndarray, float, bool, dict]):
        """Validate that step return follows the expected contract"""
        from .conftest import assert_valid_step_return
        assert_valid_step_return(step_result)
        EnvironmentValidator.validate_observation_shape(env, step_result[0])
    
    @staticmethod
    def validate_reward_bounds(reward: float, min_reward: float = -10.0, max_reward: float = 10.0):
        """Validate that reward is within reasonable bounds"""
        from .conftest import assert_reward_bounds
        assert_reward_bounds(reward, min_reward, max_reward)
    
    @staticmethod
    def validate_action_space(env: SameGameEnv, action: int):
        """Validate that action is within valid action space"""
        from .conftest import assert_valid_action_range
        max_action = env.config.num_rows * env.config.num_cols - 1
        assert_valid_action_range(action, 0, max_action)


class AgentValidator:
    """Utilities for validating agent behavior"""
    
    @staticmethod
    def validate_action_selection(agent: DqnAgent, observation: np.ndarray, expected_range: tuple[int, int]):
        """Validate that agent selects actions in expected range"""
        action = agent.act(observation)
        min_action, max_action = expected_range
        assert min_action <= action <= max_action, f"Action {action} outside range [{min_action}, {max_action}]"
        assert isinstance(action, int)
    
    @staticmethod
    def validate_epsilon_greedy_behavior(agent: DqnAgent, observation: np.ndarray, num_trials: int = 100):
        """Validate epsilon-greedy behavior through multiple action selections"""
        actions = [agent.act(observation) for _ in range(num_trials)]
        
        if agent.epsilon == 0.0:
            # Should be deterministic
            assert len(set(actions)) == 1, "Agent should be deterministic with epsilon=0"
        elif agent.epsilon == 1.0:
            # Should be mostly random (allowing for some coincidental repeats)
            unique_actions = len(set(actions))
            assert unique_actions > num_trials * 0.1, "Agent should explore with epsilon=1.0"
        else:
            # Mixed behavior - should have some randomness
            unique_actions = len(set(actions))
            assert 1 <= unique_actions <= num_trials, "Agent should show mixed behavior"
    
    @staticmethod
    def validate_learning_updates(agent: DqnAgent, initial_params: dict[str, torch.Tensor], input_shape: tuple[int, ...] | None = None):
        """Validate that learning actually updates model parameters"""
        # Use default shape if not provided
        if input_shape is None:
            input_shape = (4, 8, 8)  # Default shape
        
        # Ensure agent has enough experiences to learn
        obs = np.random.random((agent.batch_size * 2,) + input_shape).astype(np.float32)
        action_space_size = getattr(agent, 'action_space_size', 64)
        
        for i in range(agent.batch_size * 2):  # More than batch size
            agent.remember(obs[i], i % action_space_size, 1.0, obs[(i+1) % len(obs)], i == len(obs)-1)
        
        # Perform learning
        loss = agent.learn()
        
        # Check that parameters have changed
        params_changed = False
        for name, param in agent.model.named_parameters():
            if name in initial_params:
                if not torch.equal(param, initial_params[name]):
                    params_changed = True
                    break
        
        assert params_changed, "Model parameters should change after learning"
        assert isinstance(loss, torch.Tensor), "Learning should return a tensor loss"
        assert loss.item() >= 0, "Loss should be non-negative"


class TrainingValidator:
    """Utilities for validating training behavior"""
    
    @staticmethod
    def validate_training_progress(initial_epsilon: float, final_epsilon: float, epochs: int):
        """Validate that training shows expected progression"""
        epsilon_decreased = final_epsilon < initial_epsilon
        assert epsilon_decreased, "Epsilon should decrease during training"
        
        if epochs > 1:
            expected_decrease = (initial_epsilon - final_epsilon) * epochs
            # Allow for some tolerance due to minimum epsilon bounds
            assert expected_decrease > 0, "Should show epsilon progression over multiple epochs"
    
    @staticmethod
    def validate_training_results(results: list[float]):
        """Validate training results format and content"""
        assert isinstance(results, list), "Training should return a list of results"
        assert len(results) > 0, "Should have at least one result"
        assert all(isinstance(r, (int, float)) for r in results), "All results should be numeric"
        assert all(r >= 0 for r in results), "All results should be non-negative"
    
    @staticmethod
    def validate_memory_accumulation(agent: DqnAgent, initial_memory_size: int, expected_minimum_increase: int):
        """Validate that training accumulates experiences in replay buffer"""
        current_memory_size = len(agent.replay_buffer)
        memory_increase = current_memory_size - initial_memory_size
        assert memory_increase >= expected_minimum_increase, \
            f"Memory should increase by at least {expected_minimum_increase}, got {memory_increase}"


class PerformanceValidator:
    """Utilities for validating performance characteristics"""
    
    @staticmethod
    def validate_memory_usage(operation, max_memory_mb: float = 100.0):
        """Validate that an operation doesn't use excessive memory"""
        gc.collect()  # Clean up before measurement
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = operation()
        
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        assert memory_increase <= max_memory_mb, \
            f"Operation used {memory_increase:.1f}MB, expected <= {max_memory_mb}MB"
        
        return result
    
    @staticmethod
    def validate_execution_time(operation, max_time_seconds: float = 1.0):
        """Validate that an operation completes within time limit"""
        start_time = time.time()
        result = operation()
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time <= max_time_seconds, \
            f"Operation took {execution_time:.3f}s, expected <= {max_time_seconds}s"
        
        return result


# Specialized Mock Objects





# Test Data Generators with Specific Patterns

def generate_isolated_tiles_board(config: GameConfig) -> list[list[int]]:
    """Generate a board with no connected components for testing edge cases"""
    board = []
    for r in range(config.num_rows):
        row = []
        for c in range(config.num_cols):
            color = ((r + c) % (config.num_colors - 1)) + 1  # Avoid color 0
            row.append(color)
        board.append(row)
    return board




# Integration Test Helpers

def run_complete_episode(env: SameGameEnv, agent: DqnAgent, max_steps: int = 100) -> dict[str, float | int | list[int] | np.ndarray | bool]:
    """Run a complete episode and return statistics"""
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    actions_taken = []
    
    for _ in range(max_steps):
        action = agent.act(obs)
        actions_taken.append(action)
        
        obs, reward, done, _info = env.step(action)
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    return {
        'total_reward': total_reward,
        'steps': steps,
        'actions_taken': actions_taken,
        'final_observation': obs,
        'completed': done
    }


def verify_training_integration(agent_builder: AgentTestBuilder, env_builder: EnvironmentTestBuilder,
                              epochs: int = 10) -> dict[str, float | int | list[float] | bool]:
    """Verify that agent and environment integrate correctly during training"""
    from samegamerl.training.training_manager import TrainingManager

    agent = agent_builder.build()
    env = env_builder.build()

    initial_epsilon = agent.epsilon
    initial_buffer_size = len(agent.replay_buffer)

    manager = TrainingManager(agent=agent, env=env, experiment_name="test")

    results = manager.train(
        epochs=epochs,
        max_steps=20,
        report_num=2,
        visualize_num=0,
        update_target_num=epochs + 1  # Don't update target during test
    )
    
    return {
        'results': results,
        'epsilon_change': initial_epsilon - agent.epsilon,
        'buffer_growth': len(agent.replay_buffer) - initial_buffer_size,
        'training_completed': len(results) > 0
    }