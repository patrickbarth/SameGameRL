from samegamerl.agents.dqn_agent import DqnAgent
from samegamerl.environments.samegame_env import SameGameEnv


def validate(agent: DqnAgent, num_games=10) -> tuple[int, int, dict[str, list[int]]]:
    env = SameGameEnv()
    dones = 0
    terminals = 0
    left = []
    singles_left = []
    rewards = []

    agent.model.eval()
    for game in range(num_games):
        obs = env.reset()
        total_reward = 0
        for step in range(150):
            action = agent.act(obs)

            next_obs, reward, done, _ = env.step(action)

            obs = next_obs
            total_reward += reward

            if done:
                dones += 1
                left.append(0)
                singles_left.append(0)
                rewards.append(total_reward)
                break

            if env.game.get_singles() == env.game.left:
                terminals += 1
                left.append(env.game.left)
                singles_left.append(env.game.left)
                rewards.append(total_reward)
                break

        left.append(env.game.left)
        singles_left.append(env.game.get_singles())
        rewards.append(total_reward)

    return (
        dones,
        terminals,
        {
            "left": left,
            "singles_left": singles_left,
            "rewards": rewards,
        },
    )
