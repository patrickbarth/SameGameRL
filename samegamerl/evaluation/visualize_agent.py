from copy import deepcopy
from math import ceil
import pygame
import sys

from samegamerl.agents.dqn_agent import DqnAgent
from samegamerl.game.View import View
from samegamerl.game.game_config import GameFactory
from samegamerl.environments.samegame_env import SameGameEnv


def ini_visualization(game):
    pygame.init()
    view = View(game)
    screen = pygame.display.set_mode((view.screen_width, view.screen_height))
    pygame.display.set_caption("SameGame - AI playing")

    return screen, view


def play_eval_game(agent: DqnAgent, visualize=False, waiting_time=50, config=None):
    if config is None:
        config = GameFactory.default()
    env = SameGameEnv(config)
    obs = env.reset()
    rounds = 0
    if visualize:
        screen, view = ini_visualization(env.game)
    done = env.done
    agent.model.eval()
    cur_epsilon = agent.epsilon
    agent.epsilon = 0
    prev_left = env.game.left

    while (not done) and rounds < ceil(env.config.total_cells / 3):
        move, values = agent.act_visualize(obs)
        if visualize:
            screen.fill((255, 255, 255))
            board = deepcopy(env.game.get_board())
            clicked_tile = env._to_2d(move)
            board[clicked_tile[0]][clicked_tile[1]] = 6
            view.draw_text(screen, board, values.tolist())
            pygame.display.flip()
            pygame.time.wait(waiting_time)
        obs, reward, done, _ = env.step(move)
        print(reward)
        if prev_left == env.game.left:
            break
        prev_left = env.game.left
        rounds += 1
    agent.model.train()
    agent.epsilon = cur_epsilon
    pygame.quit()
    return env.game.left
