import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from samegamerl.agents.dqn_agent import DqnAgent
from samegamerl.agents.replay_buffer import ReplayBuffer


class SimpleTestModel(nn.Module):
    """Simple model for testing purposes"""
    def __init__(self, input_size=64, output_size=64):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        # Flatten input for linear layer
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        return self.linear(x_flat)


class TestDQNAgentInitialization:
    """Test DQN agent initialization and configuration"""
    
    def test_basic_initialization(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="test_model",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1
        )
        
        # Check basic attributes
        assert agent.model_name == "test_model"
        assert agent.learning_rate == 0.001
        assert agent.epsilon == 1.0
        assert agent.epsilon_decay == 0.001
        assert agent.epsilon_min == 0.1
        assert agent.batch_size == 128  # default value
        assert agent.gamma == 0.95  # default value
        assert agent.tau == 0.5  # default value
    
    def test_custom_hyperparameters(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="custom_model",
            learning_rate=0.01,
            initial_epsilon=0.8,
            epsilon_decay=0.002,
            final_epsilon=0.05,
            batch_size=256,
            gamma=0.99,
            tau=0.1
        )
        
        assert agent.batch_size == 256
        assert agent.gamma == 0.99
        assert agent.tau == 0.1
    
    def test_device_selection(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="test_model",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1
        )
        
        # Should select appropriate device
        expected_device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        assert agent.device == expected_device
    
    def test_model_and_target_model_creation(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="test_model",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1
        )
        
        # Both models should exist and be on correct device
        assert agent.model is not None
        assert agent.target_model is not None
        assert next(agent.model.parameters()).device.type == agent.device.split(':')[0]
        assert next(agent.target_model.parameters()).device.type == agent.device.split(':')[0]
    
    def test_optimizer_and_criterion_setup(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="test_model",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1
        )
        
        assert isinstance(agent.opt, torch.optim.Adam)
        assert isinstance(agent.criterion, torch.nn.MSELoss)
        
        # Check optimizer learning rate
        assert agent.opt.param_groups[0]['lr'] == 0.001
    
    def test_replay_buffer_initialization(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="test_model",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1
        )
        
        assert isinstance(agent.replay_buffer, ReplayBuffer)
        assert agent.replay_buffer.buffer.maxlen == 5000  # default capacity


class TestDQNAgentActionSelection:
    """Test action selection mechanisms"""
    
    def test_act_epsilon_greedy(self):
        model = SimpleTestModel(input_size=16, output_size=4)
        agent = DqnAgent(
            model=model,
            model_name="test_model",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1
        )
        
        observation = np.random.random((4, 2, 2)).astype(np.float32)
        
        # With epsilon=1.0, should always be random
        actions = [agent.act(observation) for _ in range(10)]
        # Actions should be in valid range
        assert all(0 <= action < 4 for action in actions)
    
    def test_act_greedy_mode(self):
        model = SimpleTestModel(input_size=16, output_size=4)
        agent = DqnAgent(
            model=model,
            model_name="test_model",
            learning_rate=0.001,
            initial_epsilon=0.0,  # No randomness
            epsilon_decay=0.001,
            final_epsilon=0.0
        )
        
        observation = np.random.random((4, 2, 2)).astype(np.float32)
        
        # Should consistently choose same action (greedy)
        actions = [agent.act(observation) for _ in range(5)]
        # All actions should be the same
        assert len(set(actions)) == 1
        assert 0 <= actions[0] < 4
    
    def test_act_visualize(self):
        model = SimpleTestModel(input_size=16, output_size=4)
        agent = DqnAgent(
            model=model,
            model_name="test_model",
            learning_rate=0.001,
            initial_epsilon=0.0,
            epsilon_decay=0.001,
            final_epsilon=0.0
        )
        
        observation = np.random.random((4, 2, 2)).astype(np.float32)
        action, q_values = agent.act_visualize(observation)
        
        assert isinstance(action, int)
        assert 0 <= action < 4
        # q_values should be a tensor or None (depending on epsilon)
        if q_values is not None:
            assert isinstance(q_values, torch.Tensor)
    
    @patch('random.random')
    def test_epsilon_thresholding(self, mock_random):
        model = SimpleTestModel(input_size=16, output_size=4)
        agent = DqnAgent(
            model=model,
            model_name="test_model",
            learning_rate=0.001,
            initial_epsilon=0.5,
            epsilon_decay=0.001,
            final_epsilon=0.1
        )
        
        observation = np.random.random((4, 2, 2)).astype(np.float32)
        
        # Test random action when random() < epsilon
        mock_random.return_value = 0.3  # < 0.5
        with patch('random.randint') as mock_randint:
            mock_randint.return_value = 2
            action = agent.act(observation)
            assert action == 2
            mock_randint.assert_called_once()
        
        # Test greedy action when random() >= epsilon
        mock_random.return_value = 0.7  # >= 0.5
        action = agent.act(observation)
        assert isinstance(action, int)


class TestDQNAgentLearning:
    """Test learning mechanisms"""
    
    def test_remember_experience(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="test_model",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1
        )
        
        # Create experience
        obs = np.random.random((4, 8, 8)).astype(np.float32)
        action = 0
        reward = 1.5
        next_obs = np.random.random((4, 8, 8)).astype(np.float32)
        done = False
        
        initial_len = len(agent.replay_buffer)
        agent.remember(obs, action, reward, next_obs, done)
        assert len(agent.replay_buffer) == initial_len + 1
    
    def test_learn_insufficient_data(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="test_model",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1,
            batch_size=32
        )
        
        # Add fewer experiences than batch size
        for i in range(10):
            obs = np.random.random((4, 8, 8)).astype(np.float32)
            agent.remember(obs, i, 0.0, obs, False)
        
        # Should return 0 (no learning)
        loss = agent.learn()
        assert loss == 0
    
    def test_learn_with_sufficient_data(self):
        model = SimpleTestModel(input_size=64, output_size=64)
        agent = DqnAgent(
            model=model,
            model_name="test_model",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1,
            batch_size=16
        )
        
        # Add sufficient experiences
        for i in range(32):
            obs = np.random.random((1, 8, 8)).astype(np.float32)
            next_obs = np.random.random((1, 8, 8)).astype(np.float32)
            agent.remember(obs, i % 64, np.random.random(), next_obs, i == 31)
        
        # Should perform learning and return loss
        loss = agent.learn()
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_decrease_epsilon(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="test_model",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.1,
            final_epsilon=0.2
        )
        
        initial_epsilon = agent.epsilon
        agent.decrease_epsilon()
        
        # Should decrease but not go below minimum
        assert agent.epsilon == initial_epsilon - 0.1
        
        # Test minimum bound
        agent.epsilon = 0.25
        agent.decrease_epsilon()
        assert agent.epsilon == 0.2  # Should stop at epsilon_min
        
        agent.decrease_epsilon()
        assert agent.epsilon == 0.2  # Should not decrease further
    
    def test_update_target_model(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="test_model",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1,
            tau=0.5
        )
        
        # Modify main model parameters
        with torch.no_grad():
            for param in agent.model.parameters():
                param.fill_(1.0)
        
        # Store original target model parameters
        original_target_params = [p.clone() for p in agent.target_model.parameters()]
        
        # Update target model
        agent.update_target_model()
        
        # Target model parameters should be updated (soft update)
        for orig_param, updated_param in zip(original_target_params, agent.target_model.parameters()):
            assert not torch.equal(orig_param, updated_param)


class TestDQNAgentPersistence:
    """Test model saving and loading"""
    
    def test_save_model(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="test_save_model",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1
        )
        
        # Test saving with default name
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory for test
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                os.makedirs("samegamerl/models", exist_ok=True)
                
                agent.save()
                
                # Check file exists
                assert os.path.exists("samegamerl/models/test_save_model.pth")
                
                # Load and verify structure
                checkpoint = torch.load("samegamerl/models/test_save_model.pth", 
                                       map_location="cpu")
                assert "model_state_dict" in checkpoint
                assert "optimizer_state_dict" in checkpoint
                assert "target_model_state_dict" in checkpoint
            finally:
                os.chdir(original_cwd)
    
    def test_save_model_custom_name(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="original_name",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                os.makedirs("samegamerl/models", exist_ok=True)
                
                agent.save("custom_name")
                
                assert os.path.exists("samegamerl/models/custom_name.pth")
            finally:
                os.chdir(original_cwd)
    
    def test_load_model(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="test_load_model",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                os.makedirs("samegamerl/models", exist_ok=True)
                
                # Modify model parameters to something recognizable
                with torch.no_grad():
                    for param in agent.model.parameters():
                        param.fill_(2.0)
                
                # Save model
                agent.save()
                
                # Create new agent and load
                new_model = SimpleTestModel()
                new_agent = DqnAgent(
                    model=new_model,
                    model_name="test_load_model",
                    learning_rate=0.002,  # Different from original
                    initial_epsilon=0.5,
                    epsilon_decay=0.002,
                    final_epsilon=0.05
                )
                
                new_agent.load()
                
                # Check that model parameters were loaded
                for orig_param, loaded_param in zip(agent.model.parameters(), 
                                                   new_agent.model.parameters()):
                    assert torch.allclose(orig_param, loaded_param)
                
                # Check that optimizer state was loaded
                assert new_agent.opt.param_groups[0]['lr'] == 0.001  # Original lr
            finally:
                os.chdir(original_cwd)
    
    def test_load_target_model(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="test_load_target",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                os.makedirs("samegamerl/models", exist_ok=True)
                
                # Modify target model
                with torch.no_grad():
                    for param in agent.target_model.parameters():
                        param.fill_(3.0)
                
                agent.save()
                
                # Load with target model
                new_model = SimpleTestModel()
                new_agent = DqnAgent(
                    model=new_model,
                    model_name="test_load_target",
                    learning_rate=0.001,
                    initial_epsilon=1.0,
                    epsilon_decay=0.001,
                    final_epsilon=0.1
                )
                
                new_agent.load(load_target=True)
                
                # Target model should match saved version
                for orig_param, loaded_param in zip(agent.target_model.parameters(),
                                                   new_agent.target_model.parameters()):
                    assert torch.allclose(orig_param, loaded_param)
            finally:
                os.chdir(original_cwd)


class TestDQNAgentIntegration:
    """Test integration scenarios"""
    
    def test_complete_training_step(self):
        model = SimpleTestModel(input_size=64, output_size=64)
        agent = DqnAgent(
            model=model,
            model_name="integration_test",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1,
            batch_size=8
        )
        
        # Simulate episode
        obs = np.random.random((4, 4, 4)).astype(np.float32)
        
        for step in range(20):
            action = agent.act(obs)
            assert 0 <= action < 64
            
            # Simulate environment response
            next_obs = np.random.random((4, 4, 4)).astype(np.float32)
            reward = np.random.uniform(-1, 1)
            done = step == 19
            
            agent.remember(obs, action, reward, next_obs, done)
            obs = next_obs
            
            # Try learning
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.learn()
                assert isinstance(loss, torch.Tensor)
            
            agent.decrease_epsilon()
        
        # Epsilon should have decreased
        assert agent.epsilon < 1.0
        
        # Should be able to update target model
        agent.update_target_model()
    
    def test_model_mode_switching(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="mode_test",
            learning_rate=0.001,
            initial_epsilon=0.0,  # Always greedy
            epsilon_decay=0.001,
            final_epsilon=0.0
        )
        
        obs = np.random.random((4, 2, 2)).astype(np.float32)
        
        # Act should switch to eval mode during inference
        initial_training_mode = agent.model.training
        action = agent.act(obs)
        final_training_mode = agent.model.training
        
        # Should return to training mode after action
        assert final_training_mode == initial_training_mode
    
    def test_different_model_architectures(self):
        """Test that agent works with different model architectures"""
        
        class CNNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(4, 8, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(8, 64)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        cnn_model = CNNModel()
        agent = DqnAgent(
            model=cnn_model,
            model_name="cnn_test",
            learning_rate=0.001,
            initial_epsilon=0.0,
            epsilon_decay=0.001,
            final_epsilon=0.0
        )
        
        # Should work with CNN input shape
        obs = np.random.random((4, 8, 8)).astype(np.float32)
        action = agent.act(obs)
        assert 0 <= action < 64


class TestDQNAgentErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_observation_shape(self):
        model = SimpleTestModel(input_size=16, output_size=4)
        agent = DqnAgent(
            model=model,
            model_name="error_test",
            learning_rate=0.001,
            initial_epsilon=0.0,
            epsilon_decay=0.001,
            final_epsilon=0.0
        )
        
        # Wrong observation shape should raise error or handle gracefully
        wrong_obs = np.random.random((2, 3, 5)).astype(np.float32)
        
        try:
            action = agent.act(wrong_obs)
            # If it doesn't raise an error, at least check the action is valid
            assert isinstance(action, int)
        except RuntimeError:
            # PyTorch runtime error due to shape mismatch is acceptable
            pass
    
    def test_empty_model_name(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1
        )
        
        # Should handle empty model name
        assert agent.model_name == ""
    
    def test_zero_learning_rate(self):
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="zero_lr_test",
            learning_rate=0.0,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1
        )
        
        # Should not crash with zero learning rate
        assert agent.learning_rate == 0.0
        assert agent.opt.param_groups[0]['lr'] == 0.0
    
    def test_extreme_hyperparameters(self):
        model = SimpleTestModel()
        
        # Test with extreme values
        agent = DqnAgent(
            model=model,
            model_name="extreme_test",
            learning_rate=1.0,  # Very high
            initial_epsilon=0.0,  # No exploration
            epsilon_decay=1.0,  # Large decay
            final_epsilon=0.0,
            batch_size=1,  # Very small batch
            gamma=0.0,  # No future rewards
            tau=1.0  # Complete target update
        )
        
        assert agent.gamma == 0.0
        assert agent.tau == 1.0
        assert agent.batch_size == 1


class TestDQNAgentCompatibility:
    """Test compatibility with base agent interface"""
    
    def test_base_agent_interface(self):
        from samegamerl.agents.base_agent import BaseAgent
        
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="compatibility_test",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1
        )
        
        # Should be instance of BaseAgent
        assert isinstance(agent, BaseAgent)
        
        # Should have required methods
        assert hasattr(agent, 'act')
        assert hasattr(agent, 'act_visualize')
        assert hasattr(agent, 'learn')
        assert hasattr(agent, 'save')
        assert hasattr(agent, 'load')
        
        # Methods should be callable
        obs = np.random.random((4, 2, 2)).astype(np.float32)
        
        action = agent.act(obs)
        assert isinstance(action, int)
        
        action, q_vals = agent.act_visualize(obs)
        assert isinstance(action, int)
        
        loss = agent.learn()  # May return 0 if insufficient data
        assert isinstance(loss, (int, float, torch.Tensor))
    
    def test_method_signatures_match_base_class(self):
        """Ensure method signatures match base class requirements"""
        model = SimpleTestModel()
        agent = DqnAgent(
            model=model,
            model_name="signature_test",
            learning_rate=0.001,
            initial_epsilon=1.0,
            epsilon_decay=0.001,
            final_epsilon=0.1
        )
        
        obs = np.random.random((4, 2, 2)).astype(np.float32)
        
        # act should take numpy array and return int
        action = agent.act(obs)
        assert isinstance(action, int)
        
        # act_visualize should return tuple
        result = agent.act_visualize(obs)
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        # learn should return something (loss or 0)
        loss = agent.learn()
        assert loss is not None