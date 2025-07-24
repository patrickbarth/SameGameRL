from samegamerl.environments.samegame_env import SameGameEnv
from samegamerl.agents.dqn_agent import DqnAgent
from samegamerl.evaluation.plot_helper import plot_results
from tqdm import tqdm

from samegamerl.evaluation.visualize_agent import play_eval_game

# boundaries
batch_size = 256
n_episodes = 20_000
max_steps = 20  # half of cells on the field is a reasonable value
update_target_freq = 10
report_freq = 100
initial_update_done = n_episodes // 2

# hyperparameters
learning_rate = 0.0001
start_epsilon = 1.0
epsilon_decay = start_epsilon / n_episodes
final_epsilon = 0.1

env = SameGameEnv()
agent = DqnAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    gamma=0,
    batch_size=batch_size,
)

total_reward = 0
loss = 0
results = []

agent.load()

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

    loss += agent.learn()
    agent.decrease_epsilon()

    if episode % update_target_freq == 0:
        agent.update_target_model()

    # if episode == initial_update_done:
    # agent.gamma = 0.8
    # agent.epsilon = 1

    if episode % report_freq == 0:
        results.append(float(loss))
        # print(f"Episode {episode+1} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f} - Loss: {loss:.5f} ")
        # play_eval_game(agent, visualize=True)
        # total_reward = 0
        loss = 0

agent.save()

plot_results(results, interval=20)
