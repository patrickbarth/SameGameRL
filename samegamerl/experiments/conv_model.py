from samegamerl.environments.samegame_env import SameGameEnv
from samegamerl.agents.dqn_agent import DqnAgent
from samegamerl.evaluation.plot_helper import plot_evals, plot_result
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from samegamerl.evaluation.validator import validate
from samegamerl.game.game_params import NUM_COLORS, NUM_ROWS, NUM_COLS

from samegamerl.evaluation.visualize_agent import play_eval_game

"""
Experiment meta info
"""


experiment_name = "CNN"


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
n_episodes = 100_000
max_steps = 20  # half of cells on the field is a reasonable value

# intervals
update_target_freq = 100
report_freq = 100
initial_update_done = n_episodes // 2
visualize_freq = 5000

# agent specific
learning_rate = 0.0001
start_epsilon = 1.0
epsilon_decay = start_epsilon / n_episodes
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

agent.load(name="CNN_base")


# used during training loop
total_reward = 0
loss = 0
results = []


"""
Training loop
"""

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
        if step > max_steps / 2 and env.game.get_singles() == env.game.left:
            break

    cur_loss = agent.learn()
    loss = (loss + cur_loss) / 2
    agent.decrease_epsilon()

    if episode % report_freq == report_freq - 1:
        results.append(float(loss))
        # print(f"Episode {episode+1} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f} - Loss: {loss:.5f} ")
        # total_reward = 0
        loss = 0

    if episode % visualize_freq == visualize_freq - 1:
        play_eval_game(agent, visualize=True, waiting_time=1000)

    if episode % update_target_freq == 0:
        agent.update_target_model()

    # if episode == initial_update_done:
    # agent.gamma = 0.8
    # agent.epsilon = 1


agent.save()

"""
Evaluation
"""

plot_result(results, interval=10)

done, terminals, evals = validate(agent)
plot_evals(evals)
print(done, terminals)
