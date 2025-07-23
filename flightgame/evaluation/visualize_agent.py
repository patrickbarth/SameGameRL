from copy import deepcopy
from math import ceil
import pygame
import sys

from flightgame.agents.dqn_agent import DqnAgent
from flightgame.game.View import View
from flightgame.game.game_params import NUM_COLS, NUM_ROWS, SCREEN_HEIGHT, SCREEN_WIDTH
from flightgame.environments.samegame_env import SameGameEnv


def ini_visualization(game):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flight Game - AI playing")

    view = View(game)

    return screen, view


def play_eval_game(agent: DqnAgent, visualize=False):
    env = SameGameEnv()
    obs = env.reset()
    rounds = 0
    if visualize:
        screen, view = ini_visualization(env.game)
    done = env.done
    agent.model.eval()
    cur_epsilon = agent.epsilon
    agent.epsilon = 0

    while (not done) and rounds < ceil(NUM_COLS * NUM_ROWS / 5):
        move, values = agent.act_eval(obs)
        if visualize:
            screen.fill((255, 255, 255))
            board = deepcopy(env.game.get_board())
            clicked_tile = env._to_2d(move)
            board[clicked_tile[0]][clicked_tile[1]] = 6
            view.draw_text(screen, board, values.tolist())
            pygame.display.flip()
            pygame.time.wait(50)
        obs, _, done, _ = env.step(move)
        rounds += 1
    agent.model.train()
    agent.epsilon = cur_epsilon
    pygame.quit()
    return env.game.left