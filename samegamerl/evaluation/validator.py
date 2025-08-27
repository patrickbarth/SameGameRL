from tqdm import tqdm
from samegamerl.agents.dqn_agent import DqnAgent
from samegamerl.environments.samegame_env import SameGameEnv


def validate(
    agent: DqnAgent, num_games=100
) -> tuple[int, int, list[tuple[int, int, int]]]:
    env = SameGameEnv()
    epsilon = agent.epsilon
    agent.epsilon = 0
    dones = 0
    terminals = 0
    results = []  # fill with (left, singles_left, rewards)

    agent.model.eval()
    for game in tqdm(range(num_games)):
        obs = env.reset()
        total_reward = 0
        finished = False
        for step in range(150):
            action = agent.act(obs)

            next_obs, reward, done, _ = env.step(action)

            obs = next_obs
            total_reward += reward

            if done:
                dones += 1
                results.append((0, 0, total_reward))
                finished = True
                break

            if env.game.get_singles() == env.game.left:
                terminals += 1
                results.append((env.game.left, env.game.left, total_reward))
                finished = True
                break
        if not finished:
            results.append((env.game.left, env.game.get_singles(), total_reward))

    agent.epsilon = epsilon
    agent.model.train()

    return (dones, terminals, results)
