from samegamerl.environments.samegame_env import SameGameEnv
from samegamerl.agents.dqn_agent import DqnAgent
from samegamerl.evaluation.plot_helper import plot_result
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from samegamerl.game.game_config import GameConfig, GameFactory

from samegamerl.evaluation.visualize_agent import play_eval_game

"""
Experiment meta info
"""


experiment_name = "pyramid"


"""
Define model for experiment
"""


class NeuralNetwork(nn.Module):
    def __init__(self, config: GameConfig):
        super().__init__()
        self.config = config
        self.flatten = nn.Flatten()
        input_size = config.num_rows * config.num_cols * config.num_colors
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, config.action_space_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


"""
Set Hyperparameters
"""
# training specific
batch_size = 512
n_episodes = 100_000
max_steps = 20  # half of cells on the field is a reasonable value

# intervals
update_target_freq = 10
report_freq = 100
initial_update_done = n_episodes // 2
visualize_freq = 5000

# agent specific
learning_rate = 0.0001
start_epsilon = 1.0
epsilon_decay = start_epsilon / n_episodes
final_epsilon = 0.1
gamma = 0


"""

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


# used during training loop
total_reward = 0
loss = 0
results = []

# agent.load()

for episode in tqdm(range(n_episodes)):
    obs = env.reset()

    for step in range(max_steps):
        action = agent.act(obs)

        next_obs, reward, done, _ = env.step(action)

        agent.remember(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward

        if done:
            break

    cur_loss = agent.learn()
    loss = (loss + cur_loss) / 2
    # agent.decrease_epsilon()

    if episode % report_freq == report_freq - 1:
        results.append(float(loss))
        # print(f"Episode {episode+1} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f} - Loss: {loss:.5f} ")
        # total_reward = 0
        loss = 0

    if episode % visualize_freq == visualize_freq - 1:
        play_eval_game(agent, visualize=True, waiting_time=500)

    # if episode % update_target_freq == 0:
    #    agent.update_target_model()

    # if episode == initial_update_done:
    # agent.gamma = 0.8
    # agent.epsilon = 1


agent.save()

plot_result(results, interval=1)
