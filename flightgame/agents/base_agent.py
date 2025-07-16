from abc import ABC, abstractmethod
import numpy as np

class BaseAgent(ABC):

    @abstractmethod
    def act(self, observation: np.ndarray) -> tuple[int, int]:
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


