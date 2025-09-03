"""
Agent implementations for SameGameRL.

This module provides both learning agents (DQN) and benchmark bots for
evaluation and comparison purposes.
"""

from .base_agent import BaseAgent
from .dqn_agent import DqnAgent
from .replay_buffer import ReplayBuffer

# Benchmark bots
from .benchmark_bot_base import BenchmarkBotBase
from .random_bot import RandomBot
from .largest_group_bot import LargestGroupBot
from .greedy_singles_bot import GreedySinglesBot

__all__ = [
    'BaseAgent',
    'DqnAgent', 
    'ReplayBuffer',
    'BenchmarkBotBase',
    'RandomBot',
    'LargestGroupBot',
    'GreedySinglesBot',
]