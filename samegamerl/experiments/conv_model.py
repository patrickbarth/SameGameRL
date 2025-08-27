from samegamerl.environments.samegame_env import SameGameEnv
from samegamerl.agents.dqn_agent import DqnAgent
from samegamerl.evaluation.plot_helper import plot_evals, plot_result
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from samegamerl.evaluation.validator import validate
from samegamerl.game.game_config import GameConfig, GameFactory

from samegamerl.evaluation.visualize_agent import play_eval_game
from samegamerl.training.train import train

"""
Experiment meta info
"""


experiment_name = "CNN_new_reward"


"""
Define model for experiment
"""


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
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, config.action_space_size)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


"""
Set Hyperparameters
"""

# training specific
batch_size = 512
n_games = 100_000
max_steps = 20  # half of cells on the field is a reasonable value

# intervals
update_target_num = 10_000
report_num = 1000
visualize_num = 20
initial_update_done = n_games // 2

# agent specific
learning_rate = 0.0001
start_epsilon = 1.0
epsilon_decay = start_epsilon / n_games
final_epsilon = 0.1
gamma = 0.8


"""
Setting up environment and agent
"""

# Use medium game configuration (8x8 with 3 colors)
config = GameFactory.medium()

env = SameGameEnv(config)
agent = DqnAgent(
    model=NeuralNetwork(config),
    model_name=experiment_name,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    input_shape=config.observation_shape,
    action_space_size=config.action_space_size,
    gamma=gamma,
    batch_size=batch_size,
)

agent.load(name="CNN_base_new_reward")


"""
Training loop
"""
results = train(
    agent,
    env,
    epochs=n_games,
    max_steps=max_steps,
    report_num=report_num,
    visualize_num=visualize_num,
    update_target_num=update_target_num,
)
agent.save()


"""
Evaluation
"""

plot_result(results, interval=10)

wins, losses, evals = validate(agent, num_games=1000)
plot_evals(evals)
print(wins, losses)
