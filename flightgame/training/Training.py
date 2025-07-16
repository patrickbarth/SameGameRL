import time
from math import ceil

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pygame
import torch
from matplotlib import style
from copy import copy, deepcopy

from agents.DQN_base_bot import DQNBot_base
from flightgame.agents.DQN_bot_old import DQNBot
from game.View import View
from flightgame.game.game import Game
from game.game_params import NUM_COLORS, NUM_ROWS, NUM_COLS, SCREEN_WIDTH, SCREEN_HEIGHT


def test_game(bot, visualize=False):
    game = Game()
    rounds = 0
    if visualize:
        screen, view = ini_visualization(game)
    while not game.done() and rounds < ceil(NUM_COLS*NUM_ROWS/3):
        move, values = bot.play_test_text(game)
        if visualize:
            screen.fill((255, 255, 255))
            board = deepcopy(game.get_board())
            clicked_tile = game.movable(move)
            board[clicked_tile[0]][clicked_tile[1]] = 6
            view.draw_text(screen, board, values)
            pygame.display.flip()
            pygame.time.wait(500)
        game.rl_move(move)
        rounds += 1
    return game.left

def ini_visualization(game):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flight Game - AI playing")

    view = View(game)

    return screen, view

def test_model(bot):
    results = []
    for i in range(20):
        results.append(test_game(bot))
    return np.average(results)

def balanced_results(results, interval=10):
    balanced = []
    for i in range(len(results) - interval):
        balanced.append(np.average(results[i:i + interval]))
    return balanced

def search_lr():
    style.use('fivethirtyeight')
    fig, ax = plt.subplots()

    results = []
    for j in range(5):
        agent = DQNBot()
        agent.create_model()
        result = []
        agent.batch_size = 512
        agent.epsilon_decay = agent.epsilon_min ** (1 / 500)
        agent.gamma = 0
        learning_rate = 0.1*0.1**j
        agent.opt = torch.optim.Adam(agent.model.parameters(), lr=learning_rate)
        print("Round ", j)
        agent.epsilon = 1
        agent.train_model(32)
        for i in range(100):
            t0 = time.time()
            agent.train_model(32)
            t1 = time.time()
            result.append(test_model(agent))
            t2 = time.time()
            print("Round " + str(i) + ", timing " + str(t1 - t0) + " " + str(
                t2 - t1) + " Bot reached an average score of " + str(result[-1]))
        results.append(result)
    for i in range(len(results)):
        pl_result = balanced_results(results[i], interval=50)
        # shift = 0-pl_result[20]
        # for j in range(len(pl_result)):
            # pl_result[j] = pl_result[j]+shift
        ax.plot(pl_result, label=0.1*0.1**i)
    plt.legend()
    plt.show()

def increasing_gamma():
    style.use('fivethirtyeight')
    fig, ax = plt.subplots()

    agent = DQNBot()
    agent.model_name = "DQN_Model_single_base_" + str(NUM_COLORS) + "_" + str(NUM_ROWS) + "_" + str(NUM_COLS)
    agent.create_model()
    # agent.save()
    # agent.load(load_target=True)
    results = []
    # agent.model_name = "DQN_Model_Conv_base_top_" + str(NUM_COLORS) + "_" + str(NUM_ROWS) + "_" + str(NUM_COLS)

    agent.batch_size = 1024
    agent.gamma = 0
    agent.epsilon_decay = agent.epsilon_min**(1/500)

    old_won = 0
    wins = []

    # agent.train_model(2)
    # test_game(agent, True)
    for j in range(25):
        print("Round ", j)
        # agent.epsilon = 1
        for i in range(20):
            t0 = time.time()
            agent.train_model(32)
            t1 = time.time()
            results.append(test_model(agent))
            t2 = time.time()
            print("Round " + str(i) + ", timing " + str(t1 - t0) + " " + str(
                t2 - t1) + " Bot reached an average score of " + str(results[-1]))
        with torch.no_grad():
            game = Game()
            print(agent.model(torch.from_numpy(game.trainable_game()).float().unsqueeze(0)))
        # test_game(agent, True)
        print(agent.won-old_won)
        wins.append(agent.won-old_won)
        old_won = agent.won
        # agent.epsilon = 0.9
    agent.save()
    print(wins)
    ax.plot(balanced_results(results, interval=100))
    plt.show()

    # agent.batch_size = 16
    # for i in range(10):
    #     print("Round " + str(i))
    #     agent.train_model(1000)
    #     results.append(test_model(agent, bot))
    #     print("Bot reached an average score of " + str(results[-1]))
    #     print(results)

# increasing_gamma()
search_lr()