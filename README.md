# SameGameRL: Deep Reinforcement Learning for SameGame Puzzle

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/package%20manager-Poetry-blue)](https://python-poetry.org/)
[![Tests](https://github.com/patrickbarth/SameGameRL/workflows/Tests/badge.svg)](https://github.com/patrickbarth/SameGameRL/actions)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-modern%20python-blue)](https://docs.python.org/3/)
[![AI Assisted](https://img.shields.io/badge/AI%20Assisted-Claude%20Code-purple)](https://claude.ai/code)

## Overview

This project trains a reinforcement learning agent to completely clear SameGame puzzle boards using Deep Q-Networks (DQN).

## Learning Goals

This project is a Breakable Toy (see https://ilango.hashnode.dev/learning-by-building-breakable-toys). The primary purpose of this project is to deepen my understanding and intuition about **Reinforcement Learning** and **DQN** in particular, while practicing to write clean, maintainable code and build scalable ML infrastructure.

## Background Story

The inspiration for this project came from **Lufthansa's in-flight entertainment system**, which features a SameGame-style puzzle with multiple difficulty levels. The final challenge is a **15x15 board with 6 colors** that must be completely cleared.

During two trans-atlantic flights, my wife and I tried to solve this last level counteless times, but couldn't solve it. So, I decided to start this project and leverage reinforcement learning to beat this challenge.

## What is SameGame?

SameGame is a tile-clearing puzzle where players click on connected groups of same-colored tiles to remove them. When tiles are removed, remaining tiles fall down due to gravity, and columns shift left to fill gaps. The game has a long history of countless implementations and rule sets (see https://en.wikipedia.org/wiki/SameGame). Most versions allow the player to collect points depending on how lage the chunks of blocks are that they remove with each turn. Clearing the whole board though, turns out to be quite complex problem (see https://erikdemaine.org/papers/Clickomania_MOVES2015/paper.pdf).

<div align="center">
  <img src="https://github.com/patrickbarth/SameGameRL/blob/master/resources/Game%20Play.gif" alt="SameGame Gameplay" width="50%">
</div>

**Why SameGame is Computationally Challenging:**
- **Massive State Space**: A 15x15 board with 6 colors has 6^225 ≈ 10^175 possible states
- **Sequential Dependencies**: Each move fundamentally changes the available future moves
- **Sparse Rewards**: There are no clear valuable intermediate goals other than clearing the board
- **Long-term Planning**: Optimal play requires considering consequences 50+ moves ahead
- **No Obvious Heuristics**: Unlike chess, there are no clear piece values or positional advantages

## What is DQN?

Deep Q-Networks (DQN) combine Q-learning with deep neural networks to handle large state spaces that would be impossible for traditional tabular methods. DQN learns to approximate the optimal Q-function Q*(s,a), which represents the maximum expected future reward from taking action 'a' in state 's'.

## Project Structure

The project is organized into these components:

**Game Engine** (`samegamerl/game/`) - Pure game logic independent of RL concerns. Can simulate games with any size of boards or number of colors and provides useful helper functions, e.g. to calculate the number of isolated cells left on the board. Includes also an interactive visualization of the game, which can be used to actually play the game or visualize the agent playing the game.

**RL Environment** (`samegamerl/environments/`) - Bridges the game engine and RL algorithms using OpenAI Gym-style interface. Converts game states to neural network-friendly one-hot encoded tensors and implements reward function.

**Agents** (`samegamerl/agents/`) - DQN implementation with experience replay buffer and modifiable hyper-parameters. Modular design allows plugging in different neural network architectures.

**Training Pipeline** (`samegamerl/training/`) - Orchestrates the learning process with configurable epochs, evaluation periods, and model checkpointing. Includes epsilon decay scheduling and performance monitoring.

**Evaluation** (`samegamerl/evaluation/`) - Tools for evaluating the agent and visualizing training progress. Includes different ways to meassure an agents performance.

**Experiments** (`samegamerl/experiments/`) - Self-contained experiments in Jupyter Notebooks combining model architectures, hyperparameters, and training configurations. Enables systematic comparison of different approaches.


## Technical Infrastructure

**Core Technologies:**
- **Python 3.13+**: The whole project is written in Python and uses newer features like the built-in type hinting
- **PyTorch**: Deep learning framework for neural network implementation
- **Pygame**: for simulating and visualizing the game

**Development Tools:**
- **Poetry**: dependency management with lock files
- **pytest**: Comprehensive test suite with 90%+ code coverage
- **GitHub Actions**: Automated CI/CD testing on every commit


## Code Style and Development Philosophy

This project follows **Test-Driven Development (TDD)** principles with clean, maintainable code that can be easily extended and modified. The focus is on enabling easy experimentation and clear visualization of training progress.

**Key Principles:**
- **Write failing tests first**, implement minimal code to pass, then refactor
- **Self-documenting code** with descriptive names that eliminate need for comments
- **Type safety** throughout with comprehensive type hints
- **Modular design** allowing independent testing and modification of components
- **Configuration over constants** for maximum experimental flexibility

The codebase prioritizes **readability and maintainability** over clever optimizations, making it easy to understand, debug, and extend for future experiments.

**Development Approach:** This project was developed using **Claude Code**. While AI helped with code generation, refactoring, testing, and documentation, all architectural decisions, algorithm selection, and problem-solving approaches were human-driven.

## Key Experiments

The core experimental parameters that significantly affect training performance are:

**Neural Network Architecture:**
Combinations of convolutional layers and fully-connected layers as well as balancing Network depth vs width.

**Reward Function Design:**
Ideally the agent would only be rewarded for clearing the board and punished if left with a board of only isolated cells, but practically it is really hard to clear the board, even for relatively simple setups (see above). So, we need to give small rewards along the way that are hopefully helping the agent at some point to clear the board. On the other side these rewards need to be small enough for the agent to clear the board and not settle for less optimal solutions. So, the basic idea is to build a reward function based on 
- **Tile removal rewards**: Immediate feedback for progress
- **Singles penalty**: Discourage creating isolated tiles
- **Completion bonus**: Large reward for clearing entire board
and make sure to properly balance these difference aspects.

**DQN Hyperparameters:**
DQN appears to not be very stable, therefore these hyperparameters need to be carefully tuned in order to ensure fast-learning without getting stuck at sub-optimal strategies.
- **Learning rate (α)**: Controls gradient descent step size
- **Discount factor (γ)**: How much agent values future rewards
- **Epsilon decay**: Exploration vs exploitation schedule
- **Replay buffer size**: Memory capacity for experience storage
- **Target network update frequency**: Training stability parameter
- **Batch size**: Number of experiences per training step


Through different experiments in the `/experiments` directory these parameters are carefully tuned and compared.


## Quick Start

> **Note**: This section is currently under review and needs testing/verification before the project is finalized.

### Installation

```bash
# Clone repository
git clone https://github.com/patrickbarth/SameGameRL.git
cd SameGameRL

# Install dependencies with Poetry (recommended)
poetry install
poetry shell

# Alternative: pip installation
pip install -r requirements.txt
```

### Training a New Model

```python
from samegamerl.game.game_config import GameFactory
from samegamerl.environments.samegame_env import SameGameEnv
from samegamerl.experiments.conv_model import ConvModel
from samegamerl.training.train import train

# Configure game environment
config = GameFactory.medium()  # 8x8 board, 3 colors
env = SameGameEnv(config)

# Train CNN-based DQN
model = ConvModel(config)
train(model, env, epochs=1000, save_path="models/my_model.pth")
```

### Evaluating Trained Models

```python
from samegamerl.evaluation.validator import evaluate_model
from samegamerl.evaluation.visualize_agent import visualize_agent

# Batch evaluation
results = evaluate_model("models/CNN.pth", num_games=100)
print(f"Average score: {results['mean_score']:.2f}")

# Interactive visualization
visualize_agent("models/CNN.pth", config)
```

### Running Tests

```bash
pytest samegamerl/tests/          # Full test suite
pytest samegamerl/tests/test_dqn_agent.py  # Specific component
```

### Interactive Demo

```bash
# Launch pygame interface
python samegamerl/main.py

# Watch trained agent play
python -c "
from samegamerl.evaluation.visualize_agent import visualize_agent
from samegamerl.game.game_config import GameFactory
visualize_agent('models/CNN.pth', GameFactory.medium())
"
```

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute this code for both personal and commercial purposes.

## Future Improvements

As this is an ongoing project and I still did not manage to train an agent to reliably empty the original 15x15 board, I plan to implement the following improvements in the future:

### 🚀 Performance Optimization
- **Separate Training and Simulation**: Currently, training and game simulation run sequentially, with simulation being the bottleneck. Moving to separate optimized containers for training, inference, and game simulation would enable much faster and more efficient training.

### 🗄️ Database Integration
- **Experience and Model State Storage**: RL training is highly unstable and model performance sometimes decreases during training. A database would allow storing different model checkpoints throughout training for comparison and rollback.
- **Hard Game Repository**: Create a curated set of challenging game states that the agent struggles with, focusing training on these difficult scenarios for targeted improvement.

### 📊 Enhanced Evaluation and Visualization
- **Streamlined Metrics**: Currently measuring cells remaining, training loss, average reward, and isolated cell count. Need unified comparison framework for easier agent evaluation.
- **Advanced Visualization**: 
  - Watch multiple agents play simultaneously
  - Compare different agent versions on identical games
  - Real-time training progress visualization
- **Training History Preservation**: Store and compare results across multiple training sessions instead of losing progress between runs.

### 🎓 Curriculum Learning
- **Progressive Gamma Scaling**: Start training with γ=0 (no future rewards) to learn basic reward function, then gradually increase gamma so the agent learns to plan further ahead.

### 🔄 Multi-Scale Training
- **Transfer Learning**: Train agents on 15x15 boards but enable them to play smaller board sizes effectively, leveraging learned spatial patterns across different scales.

### 🧠 Advanced RL Techniques
- **Algorithm Diversity**: Implement agents using different RL approaches (Actor-Critic, PPO, Rainbow DQN).