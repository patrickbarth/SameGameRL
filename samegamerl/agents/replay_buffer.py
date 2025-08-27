import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state.copy(), action, reward, next_state.copy(), done))

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray, np.ndarray]:
        if batch_size == 0:
            # Return empty arrays with appropriate shapes
            return (
                np.array([], dtype=np.float32).reshape(0, *self.buffer[0][0].shape) if len(self.buffer) > 0 else np.array([], dtype=np.float32),
                (),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32).reshape(0, *self.buffer[0][3].shape) if len(self.buffer) > 0 else np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            actions,
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
