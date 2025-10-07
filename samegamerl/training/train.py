from samegamerl.environments.samegame_env import SameGameEnv
from samegamerl.agents.dqn_agent import DqnAgent
from tqdm import tqdm

# from samegamerl.evaluation.visualize_agent import play_eval_game

# NUM_COLS and NUM_ROWS are no longer used - dimensions come from environment config


def train(
    agent: DqnAgent,
    env: SameGameEnv,
    epochs: int = 1000,
    training_loops: int = 5,
    max_steps=None,  # If None, defaults to half of total cells
    report_num: int = 500,
    visualize_num: int = 10,
    update_target_num: int = 1000,
    warmup_episodes: int | None = None,
):

    if max_steps is None:
        max_steps = env.config.total_cells // 2

    report_freq = max(1, epochs // report_num)
    update_target_freq = max(1, epochs // update_target_num)

    if visualize_num == 0:
        visualize_freq = epochs + 1
    else:
        visualize_freq = max(1, epochs // visualize_num)

    # Warm up phase to create sufficient samples in the replay_buffer
    if warmup_episodes is None:
        if max_steps > 0:
            warmup_episodes = max(1, (agent.batch_size // max_steps) * 2)
        else:
            warmup_episodes = 0

    for i in range(warmup_episodes):
        obs = env.reset()

        for step in range(max_steps):
            action = agent.act(obs)

            next_obs, reward, done, _ = env.step(action)

            agent.remember(obs, action, reward, next_obs, done)
            obs = next_obs

            if done:
                break
            if step > max_steps / 2 and env.game.get_singles() == env.game.left:
                break

    total_reward = 0
    loss = 0
    results = []
    agent.model.train()

    for episode in tqdm(range(epochs)):
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

        for i in range(training_loops):
            agent.learn()

        agent.decrease_epsilon()

        if episode % report_freq == report_freq - 1:
            results.append(float(loss))
            # print(f"Episode {episode+1} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f} - Loss: {loss:.5f} ")
            # total_reward = 0
            loss = 0

        # if episode % visualize_freq == visualize_freq - 1:
        #    play_eval_game(agent, visualize=True, waiting_time=1000)

        if episode % update_target_freq == 0:
            agent.update_target_model()

    return results
