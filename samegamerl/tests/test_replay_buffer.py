import pytest
import numpy as np
from collections import deque
from samegamerl.agents.replay_buffer import ReplayBuffer


class TestReplayBufferInitialization:
    """Test replay buffer initialization and basic properties"""

    def test_initialization_with_capacity(self):
        buffer = ReplayBuffer(capacity=1000)
        assert len(buffer) == 0
        assert isinstance(buffer.buffer, deque)
        assert buffer.buffer.maxlen == 1000

    def test_initialization_with_different_capacities(self):
        for capacity in [1, 10, 100, 5000]:
            buffer = ReplayBuffer(capacity=capacity)
            assert buffer.buffer.maxlen == capacity
            assert len(buffer) == 0

    def test_initialization_with_zero_capacity(self):
        # Should work but be essentially useless
        buffer = ReplayBuffer(capacity=0)
        assert buffer.buffer.maxlen == 0
        assert len(buffer) == 0


class TestReplayBufferOperations:
    """Test basic buffer operations"""

    def test_add_single_experience(self):
        buffer = ReplayBuffer(capacity=100)

        state = np.array([[[1, 0], [0, 1]]], dtype=np.float32)
        action = 0
        reward = 1.5
        next_state = np.array([[[0, 1], [1, 0]]], dtype=np.float32)
        done = False

        buffer.add(state, action, reward, next_state, done)

        assert len(buffer) == 1

    def test_add_multiple_experiences(self):
        buffer = ReplayBuffer(capacity=100)

        for i in range(10):
            state = np.random.random((4, 8, 8)).astype(np.float32)
            action = i % 64
            reward = float(i)
            next_state = np.random.random((4, 8, 8)).astype(np.float32)
            done = i == 9

            buffer.add(state, action, reward, next_state, done)

        assert len(buffer) == 10

    def test_add_experiences_beyond_capacity(self):
        buffer = ReplayBuffer(capacity=5)

        # Add more experiences than capacity
        for i in range(10):
            state = np.random.random((2, 2, 2)).astype(np.float32)
            buffer.add(state, i, float(i), state, False)

        # Should only keep the most recent 5
        assert len(buffer) == 5

        # Should be able to sample all 5
        batch = buffer.sample(5)
        assert len(batch[1]) == 5  # actions tuple length

    def test_fifo_behavior(self):
        buffer = ReplayBuffer(capacity=3)

        # Add experiences with identifiable actions
        for action in [10, 20, 30]:
            state = np.zeros((1, 1, 1), dtype=np.float32)
            buffer.add(state, action, 0.0, state, False)

        # Add one more, should remove the first (action=10)
        buffer.add(
            np.zeros((1, 1, 1), dtype=np.float32),
            40,
            0.0,
            np.zeros((1, 1, 1), dtype=np.float32),
            False,
        )

        # Sample all and check that 10 is not present
        batch = buffer.sample(3)
        actions = batch[1]
        assert 10 not in actions
        assert 20 in actions
        assert 30 in actions
        assert 40 in actions


class TestReplayBufferSampling:
    """Test sampling functionality"""

    def test_sample_basic(self):
        buffer = ReplayBuffer(capacity=100)

        # Add some experiences
        for i in range(20):
            state = np.random.random((3, 4, 4)).astype(np.float32)
            action = i
            reward = float(i * 0.1)
            next_state = np.random.random((3, 4, 4)).astype(np.float32)
            done = i == 19

            buffer.add(state, action, reward, next_state, done)

        # Sample a batch
        batch_size = 10
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        # Check return types and shapes
        assert isinstance(states, np.ndarray)
        assert isinstance(actions, tuple)
        assert isinstance(rewards, np.ndarray)
        assert isinstance(next_states, np.ndarray)
        assert isinstance(dones, np.ndarray)

        # Check batch size
        assert len(actions) == batch_size
        assert states.shape[0] == batch_size
        assert rewards.shape[0] == batch_size
        assert next_states.shape[0] == batch_size
        assert dones.shape[0] == batch_size

        # Check data types
        assert rewards.dtype == np.float32
        assert dones.dtype == np.float32

    def test_sample_shapes_preserved(self):
        buffer = ReplayBuffer(capacity=50)
        state_shape = (4, 8, 8)  # Common shape for SameGame

        for i in range(10):
            state = np.random.random(state_shape).astype(np.float32)
            next_state = np.random.random(state_shape).astype(np.float32)
            buffer.add(state, i, 1.0, next_state, False)

        states, _, _, next_states, _ = buffer.sample(5)

        # States should be stacked properly
        assert states.shape == (5,) + state_shape
        assert next_states.shape == (5,) + state_shape

    def test_sample_randomness(self):
        buffer = ReplayBuffer(capacity=100)

        # Add experiences with distinct actions
        for i in range(50):
            state = np.zeros((2, 2, 2), dtype=np.float32)
            buffer.add(state, i, 0.0, state, False)

        # Sample multiple times and check we get different results
        sample1 = set(buffer.sample(10)[1])  # actions
        sample2 = set(buffer.sample(10)[1])  # actions

        # Should be different (with very high probability)
        assert sample1 != sample2

    def test_sample_without_replacement_within_batch(self):
        buffer = ReplayBuffer(capacity=10)

        # Add exactly 10 unique experiences
        for i in range(10):
            state = np.zeros((1, 1, 1), dtype=np.float32)
            buffer.add(state, i, 0.0, state, False)

        # Sample all 10
        _, actions, _, _, _ = buffer.sample(10)

        # Should have exactly 10 unique actions
        assert len(set(actions)) == 10
        assert len(actions) == 10


class TestReplayBufferEdgeCases:
    """Test edge cases and error conditions"""

    def test_sample_more_than_available(self):
        buffer = ReplayBuffer(capacity=100)

        # Add only 5 experiences
        for i in range(5):
            state = np.zeros((1, 1, 1), dtype=np.float32)
            buffer.add(state, i, 0.0, state, False)

        # Try to sample more than available
        with pytest.raises(ValueError):
            buffer.sample(10)

    def test_sample_from_empty_buffer(self):
        buffer = ReplayBuffer(capacity=100)

        with pytest.raises(ValueError):
            buffer.sample(1)

    def test_sample_zero_batch_size(self):
        buffer = ReplayBuffer(capacity=100)

        # Add one experience
        state = np.zeros((1, 1, 1), dtype=np.float32)
        buffer.add(state, 0, 0.0, state, False)

        # Sample zero should work and return empty arrays
        states, actions, rewards, next_states, dones = buffer.sample(0)

        assert len(actions) == 0
        assert states.shape[0] == 0
        assert rewards.shape[0] == 0
        assert next_states.shape[0] == 0
        assert dones.shape[0] == 0

    def test_capacity_one_buffer(self):
        buffer = ReplayBuffer(capacity=1)

        # Add first experience
        state1 = np.ones((2, 2, 2), dtype=np.float32)
        buffer.add(state1, 1, 1.0, state1, False)
        assert len(buffer) == 1

        # Add second experience, should replace first
        state2 = np.zeros((2, 2, 2), dtype=np.float32)
        buffer.add(state2, 2, 2.0, state2, True)
        assert len(buffer) == 1

        # Sample should return the second experience
        states, actions, rewards, next_states, dones = buffer.sample(1)
        assert actions[0] == 2
        assert rewards[0] == 2.0
        assert dones[0] == 1.0  # True converted to float
        assert np.array_equal(states[0], state2)


class TestReplayBufferDataIntegrity:
    """Test that data is stored and retrieved correctly"""

    def test_data_preservation(self):
        buffer = ReplayBuffer(capacity=100)

        # Create distinct, identifiable data
        original_state = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
        original_action = 42
        original_reward = 3.14
        original_next_state = np.array([[[5, 6], [7, 8]]], dtype=np.float32)
        original_done = True

        buffer.add(
            original_state,
            original_action,
            original_reward,
            original_next_state,
            original_done,
        )

        # Sample and verify data integrity
        states, actions, rewards, next_states, dones = buffer.sample(1)

        assert np.array_equal(states[0], original_state)
        assert actions[0] == original_action
        assert rewards[0] == original_reward
        assert np.array_equal(next_states[0], original_next_state)
        assert dones[0] == 1.0  # True becomes 1.0

    def test_different_state_shapes_handled(self):
        buffer = ReplayBuffer(capacity=100)

        # Add experiences with different but compatible shapes
        shapes = [(2, 3, 3), (2, 3, 3), (2, 3, 3)]  # Same shape for stacking

        for i, shape in enumerate(shapes):
            state = np.random.random(shape).astype(np.float32)
            buffer.add(state, i, float(i), state, i == 2)

        # Sample should work and preserve shapes
        states, actions, rewards, next_states, dones = buffer.sample(3)

        for i in range(3):
            assert states[i].shape == shapes[i]
            assert next_states[i].shape == shapes[i]

    def test_reward_and_done_conversion(self):
        buffer = ReplayBuffer(capacity=10)

        test_cases = [
            (0, False),
            (1, True),
            (-1.5, False),
            (100.0, True),
            (0.001, False),
        ]

        for reward, done in test_cases:
            state = np.zeros((1, 1, 1), dtype=np.float32)
            buffer.add(state, 0, reward, state, done)

        # Sample all and verify conversions
        _, _, rewards, _, dones = buffer.sample(len(test_cases))

        for i, (expected_reward, expected_done) in enumerate(test_cases):
            assert rewards[i] == expected_reward
            assert dones[i] == (1.0 if expected_done else 0.0)


class TestReplayBufferPerformance:
    """Test performance characteristics"""

    def test_large_capacity_initialization(self):
        # Should handle large capacities without issues
        large_capacity = 100000
        buffer = ReplayBuffer(capacity=large_capacity)
        assert buffer.buffer.maxlen == large_capacity
        assert len(buffer) == 0

    def test_frequent_additions(self):
        buffer = ReplayBuffer(capacity=1000)

        # Add many experiences rapidly
        for i in range(2000):  # More than capacity
            state = np.random.random((2, 2, 2)).astype(np.float32)
            buffer.add(state, i, 0.0, state, False)

        # Should maintain capacity limit
        assert len(buffer) == 1000

        # Should still be able to sample
        batch = buffer.sample(100)
        assert len(batch[1]) == 100

    def test_large_batch_sampling(self):
        buffer = ReplayBuffer(capacity=5000)

        # Fill buffer
        for i in range(5000):
            state = np.random.random((3, 4, 4)).astype(np.float32)
            buffer.add(state, i, 0.0, state, False)

        # Sample large batches
        for batch_size in [100, 500, 1000]:
            batch = buffer.sample(batch_size)
            assert len(batch[1]) == batch_size
            assert batch[0].shape[0] == batch_size


class TestReplayBufferIntegration:
    """Test integration with realistic RL scenarios"""

    def test_typical_samegame_usage(self):
        buffer = ReplayBuffer(capacity=5000)

        # Simulate typical SameGame RL interaction
        num_colors, num_rows, num_cols = 4, 8, 8
        state_shape = (num_colors, num_rows, num_cols)

        # Simulate episode data
        episode_length = 20
        for step in range(episode_length):
            state = np.random.randint(0, 2, state_shape).astype(np.float32)
            action = np.random.randint(0, num_rows * num_cols)
            reward = np.random.uniform(-1, 1)
            next_state = np.random.randint(0, 2, state_shape).astype(np.float32)
            done = step == episode_length - 1

            buffer.add(state, action, reward, next_state, done)

        # Should be able to sample training batches
        if len(buffer) >= 16:  # Typical batch size
            batch = buffer.sample(16)
            states, actions, rewards, next_states, dones = batch

            # Verify realistic constraints
            assert states.shape == (16, num_colors, num_rows, num_cols)
            assert all(0 <= action < num_rows * num_cols for action in actions)
            assert all(
                -10 <= reward <= 10 for reward in rewards
            )  # Reasonable reward bounds

    def test_empty_full_cycle(self):
        buffer = ReplayBuffer(capacity=10)

        # Start empty
        assert len(buffer) == 0

        # Fill to capacity
        for i in range(10):
            state = np.zeros((1, 1, 1), dtype=np.float32)
            buffer.add(state, i, 0.0, state, False)
        assert len(buffer) == 10

        # Overfill (should maintain capacity)
        for i in range(10, 20):
            state = np.zeros((1, 1, 1), dtype=np.float32)
            buffer.add(state, i, 0.0, state, False)
        assert len(buffer) == 10

        # Sample everything multiple times
        for _ in range(5):
            batch = buffer.sample(10)
            assert len(batch[1]) == 10


class TestReplayBufferRobustness:
    """Test robustness to edge cases and potential issues"""

    def test_numpy_array_copying(self):
        buffer = ReplayBuffer(capacity=10)

        # Create mutable state
        state = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
        original_state = state.copy()

        buffer.add(state, 0, 0.0, state, False)

        # Modify original state
        state[0, 0, 0] = 999

        # Buffer should have original values
        sampled_states, _, _, _, _ = buffer.sample(1)
        assert np.array_equal(sampled_states[0], original_state)
        assert sampled_states[0][0, 0, 0] != 999

    def test_mixed_data_types(self):
        buffer = ReplayBuffer(capacity=10)

        # Test various numeric types for rewards and actions
        test_data = [
            (np.int32(0), np.float64(1.5)),
            (np.int64(1), np.float32(2.5)),
            (int(2), float(3.5)),
            (np.int16(3), np.float16(4.5)),
        ]

        for action, reward in test_data:
            state = np.zeros((1, 1, 1), dtype=np.float32)
            buffer.add(state, action, reward, state, False)

        # Should handle all types and convert appropriately
        _, actions, rewards, _, _ = buffer.sample(len(test_data))

        # All should be converted to appropriate types
        assert all(isinstance(action, (int, np.integer)) for action in actions)
        assert rewards.dtype == np.float32
