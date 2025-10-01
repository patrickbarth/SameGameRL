from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    model_name: str

    @abstractmethod
    def act(self, observation: np.ndarray) -> int:
        pass

    @abstractmethod
    def act_visualize(self, observation: np.ndarray) -> tuple[int, np.ndarray]:
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass
