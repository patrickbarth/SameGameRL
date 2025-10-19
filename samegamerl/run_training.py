from samegamerl.environments.samegame_env import SameGameEnv
from samegamerl.agents.dqn_agent import DqnAgent
from samegamerl.evaluation.plot_helper import plot_evals, plot_result
from samegamerl.evaluation.benchmark import Benchmark
from tqdm import tqdm
from torch import nn
from samegamerl.game.game_config import GameConfig, GameFactory
from samegamerl.agents.replay_buffer import ReplayBuffer
from samegamerl.evaluation.benchmark_scripts import (
    _compute_stats,
    benchmark_agent,
    get_agent_performance,
)
from samegamerl.training.training_manager import TrainingManager
from samegamerl.training.checkpoint_service import CheckpointService
from samegamerl.training.pickle_checkpoint_repository import PickleCheckpointRepository
import matplotlib.pyplot as plt


experiment_name = "CNN_10"


class NeuralNetwork(nn.Module):
    def __init__(self, config: GameConfig):
        super().__init__()
        self.config = config
        self.conv_stack = nn.Sequential(
            nn.Conv2d(config.num_colors, 64, 3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d((2,2), (2,2)),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d((2,2), (2,2)),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64 * config.num_cols * config.num_rows, 512),
            nn.ReLU(),
            nn.Linear(512, config.action_space_size),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        # x = self.global_pool(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Training specific parameters
batch_size = 32
n_games = 100
max_steps = 30  # Maximum steps per episode
training_loops = 2
rounds = 2

# Training intervals
update_target_num = 2  # Target network update frequency
report_num = 100  # Progress reporting interval

# Agent hyperparameters
learning_rate = 0.001
start_epsilon = 1.0  # Initial exploration rate
epsilon_decay = start_epsilon / (n_games * rounds)
final_epsilon = 0.1  # Minimum exploration rate
gamma = 0.95  # Discount factor
tau = 1.00


# Use medium game configuration (8x8 board with 3 colors)
config = GameFactory.medium()

# Initialize environment and agent
env = SameGameEnv(config, partial_completion_base=5)
agent = DqnAgent(
    model=NeuralNetwork(config),
    config=config,
    model_name=experiment_name,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    gamma=gamma,
    batch_size=batch_size,
)

agent.replay_buffer = ReplayBuffer(capacity=50_000)

checkpoint_repository = PickleCheckpointRepository("samegamerl/experiments/checkpoints")
checkpoint_service = CheckpointService(repository=checkpoint_repository)

trainer = TrainingManager(agent, env, experiment_name, checkpoint_service)

progress = dict()
training_error = []

trainer.warmup()
trainer.train(
    epochs=n_games,
    max_steps=max_steps,
    report_num=report_num,
    update_target_num=update_target_num,
    training_loops=training_loops,
)
checkpoint_id = trainer.create_checkpoint()
results = get_agent_performance(agent, config, 20)
comparison = results["avg_tiles_cleared"]

for i in range(rounds):
    training_error += trainer.train(
        epochs=n_games,
        max_steps=max_steps,
        report_num=report_num,
        update_target_num=update_target_num,
        training_loops=training_loops,
    )

    results = get_agent_performance(agent, config, 20)

    # if results["avg_tiles_cleared"] < comparison:
    #  d  print(f"Model performance not satisfactory ({comparison} vs. {results['avg_tiles_cleared']}), rolling back last checkpoint...")
    #    trainer.rollback_to_checkpoint(checkpoint_id)
    #    trainer.warmup()
    #    continue

    checkpoint_id = trainer.create_checkpoint()
    comparison = results["avg_tiles_cleared"]

    for metric in results:
        if metric not in progress:
            progress[metric] = [results[metric]]
        else:
            progress[metric].append(results[metric])

fig, axs = plt.subplots(4, figsize=(10, 15), sharex=True)
metrics = list(progress.keys())
for i in range(len(metrics)):
    axs[i].plot(
        list(range(n_games, n_games * (len(progress[metrics[i]]) + 1), n_games)),
        progress[metrics[i]],
    )
    axs[i].set_title(metrics[i])
