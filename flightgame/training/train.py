from flightgame.environments.samegame_env import SameGameEnv
from flightgame.agents.dqn_agent import DqnAgent
from tqdm import tqdm

# boundaries
batch_size = 128
n_episodes = 10_000
max_steps = 50 # half of cells on the field is a reasonable value
update_target_freq = 10
report_freq = 50

# hyperparameters
learning_rate = 0.01
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

env = SameGameEnv()
agent = DqnAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

for episode in tqdm(range(n_episodes)):
    obs = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = agent.act(obs)

        next_obs, reward, done, _ = env.step(action)
        
        agent.remember(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward

        if done:
            break

    agent.learn()
    agent.decrease_epsilon()

    if episode % update_target_freq == 0:
        agent.update_target_model()

    if episode % report_freq == 0:
        print(f"Episode {episode+1} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")




