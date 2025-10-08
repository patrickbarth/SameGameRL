import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
from tqdm import tqdm

from samegamerl.training.training_manager import TrainingManager
from samegamerl.agents.dqn_agent import DqnAgent
from samegamerl.environments.samegame_env import SameGameEnv
from samegamerl.game.game_config import GameFactory


def train(agent, env, **kwargs):
    """Wrapper function for backward compatibility with tests.

    Creates a TrainingManager and delegates to its train() method.
    """
    manager = TrainingManager(agent=agent, env=env, experiment_name="test")
    return manager.train(**kwargs)


class MockModel:
    """Mock neural network model for testing"""
    def __init__(self):
        self.training = True
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class MockAgent:
    """Mock DQN agent for testing training logic"""
    def __init__(self):
        self.model = MockModel()
        self.epsilon = 1.0
        self.batch_size = 32  # Default batch size for testing
        self.actions_taken = []
        self.learn_calls = 0
        self.target_updates = 0
        self.epsilon_decreases = 0
        
    def act(self, observation):
        action = len(self.actions_taken) % 10  # Deterministic for testing
        self.actions_taken.append(action)
        return action
    
    def remember(self, obs, action, reward, next_obs, done):
        pass  # Mock implementation
    
    def learn(self):
        self.learn_calls += 1
        return 0.5  # Mock loss value
    
    def decrease_epsilon(self):
        self.epsilon_decreases += 1
        self.epsilon = max(0.1, self.epsilon - 0.01)
    
    def update_target_model(self):
        self.target_updates += 1


class MockEnvironment:
    """Mock environment for testing training logic"""
    def __init__(self):
        self.reset_calls = 0
        self.step_calls = 0
        self.current_step = 0
        self.game = Mock()
        self.game.get_singles.return_value = 0
        self.game.left = 10
        # Add config for TrainingManager compatibility
        self.config = Mock()
        self.config.total_cells = 64  # Default for 8x8 board
        
    def reset(self):
        self.reset_calls += 1
        self.current_step = 0
        return np.random.random((4, 8, 8)).astype(np.float32)
    
    def step(self, action):
        self.step_calls += 1
        self.current_step += 1
        
        next_obs = np.random.random((4, 8, 8)).astype(np.float32)
        reward = np.random.uniform(-1, 1)
        done = self.current_step >= 10  # Episode ends after 10 steps
        info = {}
        
        return next_obs, reward, done, info


class TestTrainFunction:
    """Test the main train function"""
    
    def test_train_basic_execution(self):
        agent = MockAgent()
        env = MockEnvironment()
        
        # Test basic training execution
        results = train(
            agent=agent,
            env=env,
            epochs=5,
            max_steps=20,
            report_num=2,
            visualize_num=0,  # Disable visualization
            update_target_num=3,
            warmup_episodes=0  # Disable warmup for test consistency
        )
        
        # Check that training completed
        assert isinstance(results, list)
        assert len(results) > 0  # Should have some results
        
        # Check that environment was used
        assert env.reset_calls == 5  # One reset per epoch
        assert env.step_calls > 0  # Some steps were taken
        
        # Check that agent methods were called
        assert agent.learn_calls > 0
        assert agent.epsilon_decreases == 5  # Once per epoch
        assert len(agent.actions_taken) > 0
    
    def test_train_epochs_parameter(self):
        agent = MockAgent()
        env = MockEnvironment()
        
        epochs = 10
        results = train(
            agent=agent,
            env=env,
            epochs=epochs,
            max_steps=5,
            report_num=5,
            visualize_num=0,
            update_target_num=20,
            warmup_episodes=0  # Disable warmup for test consistency
        )
        
        # Should have reset once per epoch
        assert env.reset_calls == epochs
        
        # Should have decreased epsilon once per epoch
        assert agent.epsilon_decreases == epochs
    
    def test_train_max_steps_limit(self):
        agent = MockAgent()
        env = MockEnvironment()
        
        max_steps = 3
        results = train(
            agent=agent,
            env=env,
            epochs=2,
            max_steps=max_steps,
            report_num=1,
            visualize_num=0,
            update_target_num=10,
            warmup_episodes=0  # Disable warmup for test consistency
        )
        
        # Each episode should be limited by max_steps
        # Since mock env ends after 10 steps, max_steps should be the limit
        assert env.step_calls <= 2 * max_steps
    
    def test_train_early_termination_on_done(self):
        agent = MockAgent()
        
        # Create environment that ends episodes quickly
        class QuickDoneEnv(MockEnvironment):
            def step(self, action):
                self.step_calls += 1
                next_obs = np.random.random((4, 8, 8)).astype(np.float32)
                reward = 1.0
                done = True  # Always done after one step
                return next_obs, reward, done, {}
        
        env = QuickDoneEnv()
        
        results = train(
            agent=agent,
            env=env,
            epochs=3,
            max_steps=20,
            report_num=1,
            visualize_num=0,
            update_target_num=10,
            warmup_episodes=0  # Disable warmup for test consistency
        )
        
        # Should have exactly 3 steps (one per epoch) due to immediate termination
        assert env.step_calls == 3
    
    def test_train_reporting_intervals(self):
        agent = MockAgent()
        env = MockEnvironment()
        
        epochs = 10
        report_num = 3  # Should report 3 times total
        
        results = train(
            agent=agent,
            env=env,
            epochs=epochs,
            max_steps=5,
            report_num=report_num,
            visualize_num=0,
            update_target_num=20
        )
        
        # Should have report_num results
        assert len(results) == report_num
        
        # All results should be numeric (loss values)
        assert all(isinstance(r, (int, float)) for r in results)
    
    def test_train_target_model_updates(self):
        agent = MockAgent()
        env = MockEnvironment()
        
        epochs = 15
        update_target_num = 5  # Should update 3 times (episodes 0, 5, 10)
        
        train(
            agent=agent,
            env=env,
            epochs=epochs,
            max_steps=5,
            report_num=1,
            visualize_num=0,
            update_target_num=update_target_num
        )
        
        # Should update target model at correct intervals
        # For 15 epochs with update_target_num=5: update_target_freq = 15//5 = 3
        # Updates happen at episodes: 0, 3, 6, 9, 12 = 5 total updates
        expected_updates = 5
        assert agent.target_updates == expected_updates


class TestTrainWithRealComponents:
    """Test training with actual agent and environment classes"""
    
    @pytest.fixture
    def simple_model(self):
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self, input_size, output_size):
                super().__init__()
                self.fc = nn.Linear(input_size, output_size)
            
            def forward(self, x):
                batch_size = x.shape[0]
                x_flat = x.view(batch_size, -1)
                return self.fc(x_flat)
        
        return SimpleModel
    
    def test_train_with_real_agent_and_env(self, simple_model):
        # Create real agent and environment
        config = GameFactory.custom(2, 2, 3)
        model = simple_model(input_size=12, output_size=4)  # 3*2*2=12 inputs, 2*2=4 actions
        agent = DqnAgent(
            model=model,
            config=config,
            model_name="test_train",
            learning_rate=0.01,
            initial_epsilon=1.0,
            epsilon_decay=0.1,
            final_epsilon=0.1,
            batch_size=4
        )
        
        env = SameGameEnv(config)
        
        # Test short training run
        results = train(
            agent=agent,
            env=env,
            epochs=5,
            max_steps=8,
            report_num=2,
            visualize_num=0,
            update_target_num=10
        )
        
        # Should complete without errors
        assert isinstance(results, list)
        assert len(results) == 2  # report_num
        
        # Agent should have learned (epsilon should decrease)
        assert agent.epsilon < 1.0
        
        # Should have some experiences in replay buffer
        assert len(agent.replay_buffer) > 0
    
    def test_train_memory_accumulation(self, simple_model):
        config = GameFactory.custom(2, 2, 3)
        model = simple_model(input_size=12, output_size=4)  # 3*2*2=12 inputs, 2*2=4 actions
        agent = DqnAgent(
            model=model,
            config=config,
            model_name="memory_test",
            learning_rate=0.01,
            initial_epsilon=1.0,
            epsilon_decay=0.05,
            final_epsilon=0.1,
            batch_size=8
        )
        
        env = SameGameEnv(config)
        
        initial_buffer_size = len(agent.replay_buffer)
        
        train(
            agent=agent,
            env=env,
            epochs=10,
            max_steps=5,
            report_num=1,
            visualize_num=0,
            update_target_num=20
        )
        
        # Replay buffer should have accumulated experiences
        assert len(agent.replay_buffer) > initial_buffer_size
        
        # Should not exceed buffer capacity
        assert len(agent.replay_buffer) <= agent.replay_buffer.buffer.maxlen


class TestTrainEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_train_single_epoch(self):
        agent = MockAgent()
        env = MockEnvironment()
        
        results = train(
            agent=agent,
            env=env,
            epochs=1,
            max_steps=10,
            report_num=1,
            visualize_num=0,
            update_target_num=1,
            warmup_episodes=0  # Disable warmup for test consistency
        )
        
        # Should handle single epoch correctly
        assert len(results) == 1
        assert env.reset_calls == 1
        assert agent.epsilon_decreases == 1
    
    def test_train_zero_max_steps(self):
        agent = MockAgent()
        env = MockEnvironment()
        
        results = train(
            agent=agent,
            env=env,
            epochs=3,
            max_steps=0,  # No steps allowed
            report_num=1,
            visualize_num=0,
            update_target_num=10
        )
        
        # Should still reset environment and call agent methods
        assert env.reset_calls == 3
        assert env.step_calls == 0  # No steps taken
        assert agent.epsilon_decreases == 3
    
    def test_train_large_report_num(self):
        agent = MockAgent()
        env = MockEnvironment()
        
        epochs = 5
        report_num = 100  # More than epochs
        
        results = train(
            agent=agent,
            env=env,
            epochs=epochs,
            max_steps=5,
            report_num=report_num,
            visualize_num=0,
            update_target_num=10
        )
        
        # Should report at every epoch when report_num > epochs
        assert len(results) == epochs
    
    def test_train_large_update_target_num(self):
        agent = MockAgent()
        env = MockEnvironment()
        
        epochs = 5
        update_target_num = 100  # Larger than epochs
        
        train(
            agent=agent,
            env=env,
            epochs=epochs,
            max_steps=5,
            report_num=1,
            visualize_num=0,
            update_target_num=update_target_num
        )
        
        # Should update target model at least once (at episode 0)
        assert agent.target_updates >= 1
    
    def test_train_singles_early_termination(self):
        agent = MockAgent()
        
        # Create environment that simulates all-singles condition
        class SinglesEnv(MockEnvironment):
            def __init__(self):
                super().__init__()
                self.game = Mock()
                self.game.get_singles.return_value = 5
                self.game.left = 5  # All tiles are singles
        
        env = SinglesEnv()
        
        max_steps = 20
        results = train(
            agent=agent,
            env=env,
            epochs=2,
            max_steps=max_steps,
            report_num=1,
            visualize_num=0,
            update_target_num=10,
            warmup_episodes=0  # Disable warmup for test consistency
        )
        
        # Should terminate early due to singles condition
        # Step count should be less than 2 * max_steps due to early termination
        assert env.step_calls < 2 * max_steps


class TestTrainParameterValidation:
    """Test parameter validation and bounds"""
    
    def test_train_parameter_types(self):
        agent = MockAgent()
        env = MockEnvironment()
        
        # All parameters should work with various numeric types
        results = train(
            agent=agent,
            env=env,
            epochs=int(3),
            max_steps=np.int32(5),
            report_num=float(2),  # Should convert to int
            visualize_num=np.int64(0),
            update_target_num=int(10)
        )
        
        assert isinstance(results, list)
        assert len(results) > 0


class TestTrainProgressTracking:
    """Test progress tracking and metrics"""

    @patch('samegamerl.training.training_manager.tqdm')
    def test_train_uses_progress_bar(self, mock_tqdm):
        agent = MockAgent()
        env = MockEnvironment()
        
        # Mock tqdm to return a simple range
        mock_tqdm.return_value = range(5)
        
        train(
            agent=agent,
            env=env,
            epochs=5,
            max_steps=3,
            report_num=1,
            visualize_num=0,
            update_target_num=10
        )
        
        # Should call tqdm with range of epochs
        mock_tqdm.assert_called_once_with(range(5))
    
    def test_train_loss_accumulation(self):
        # Test that loss is properly accumulated and averaged
        class LossTrackingAgent(MockAgent):
            def __init__(self):
                super().__init__()
                self.loss_values = [1.0, 0.5, 0.3, 0.2, 0.1]
                self.loss_index = 0
            
            def learn(self):
                self.learn_calls += 1
                if self.loss_index < len(self.loss_values):
                    loss = self.loss_values[self.loss_index]
                    self.loss_index += 1
                    return loss
                return 0.1
        
        agent = LossTrackingAgent()
        env = MockEnvironment()
        
        results = train(
            agent=agent,
            env=env,
            epochs=5,
            max_steps=5,
            report_num=5,  # Report every epoch
            visualize_num=0,
            update_target_num=10
        )
        
        # Results should contain loss values
        assert len(results) == 5
        assert all(isinstance(r, (int, float)) for r in results)
        
        # Loss should generally decrease or stay reasonable
        assert all(0 <= r <= 2.0 for r in results)


class TestTrainIntegration:
    """Test integration with other components"""

    def test_train_model_mode_management(self):
        """Test that model training mode is properly managed"""
        agent = MockAgent()
        env = MockEnvironment()
        
        # Track model mode changes
        model_modes = []
        original_train = agent.model.train
        
        def track_train():
            model_modes.append('train')
            return original_train()
        
        agent.model.train = track_train
        
        train(
            agent=agent,
            env=env,
            epochs=2,
            max_steps=5,
            report_num=1,
            visualize_num=0,
            update_target_num=10
        )
        
        # Model should be set to training mode at the beginning
        assert 'train' in model_modes
    
    def test_train_with_different_environment_responses(self):
        """Test training with various environment response patterns"""
        
        class VariableRewardEnv(MockEnvironment):
            def __init__(self):
                super().__init__()
                self.rewards = [1.0, -0.5, 2.0, 0.0, -1.0, 3.0]
                self.reward_index = 0
            
            def step(self, action):
                next_obs = np.random.random((4, 8, 8)).astype(np.float32)
                reward = self.rewards[self.reward_index % len(self.rewards)]
                self.reward_index += 1
                self.step_calls += 1
                done = self.step_calls % 5 == 0  # Episode every 5 steps
                return next_obs, reward, done, {}
        
        agent = MockAgent()
        env = VariableRewardEnv()
        
        results = train(
            agent=agent,
            env=env,
            epochs=3,
            max_steps=10,
            report_num=1,
            visualize_num=0,
            update_target_num=10
        )
        
        # Should handle variable rewards without issues
        assert isinstance(results, list)
        assert len(results) >= 1
        assert agent.learn_calls > 0