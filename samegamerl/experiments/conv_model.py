from samegamerl.environments.samegame_env import SameGameEnv
from samegamerl.agents.dqn_agent import DqnAgent
from samegamerl.evaluation.plot_helper import plot_evals, plot_result
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from samegamerl.evaluation.validator import validate
from samegamerl.game.game_params import NUM_COLORS, NUM_ROWS, NUM_COLS

from samegamerl.evaluation.visualize_agent import play_eval_game
from samegamerl.training.train import train

"""
Experiment meta info
"""


experiment_name = "test"


"""
Define model for experiment
"""


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(NUM_COLORS, 64, 3, padding=1),
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
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, NUM_ROWS * NUM_COLS)
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
n_games = 10_000
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
gamma = 0.5


"""
Setting up environment and agent
"""

env = SameGameEnv()
agent = DqnAgent(
    model=NeuralNetwork(),
    model_name=experiment_name,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    gamma=gamma,
    batch_size=batch_size,
)

# agent.load(name="CNN_base")

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

# agent.save()

"""
Evaluation
"""

plot_result(results, interval=10)

wins, losses, evals = validate(agent)
plot_evals(evals)
print(wins, losses)
