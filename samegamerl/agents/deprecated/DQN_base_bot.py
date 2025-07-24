import random
from copy import deepcopy
from math import ceil

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

'''
from Game.game_logic import GameLogic
from Game.game_params import NUM_COLORS, NUM_ROWS, NUM_COLS


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(NUM_COLORS, 256, 3),
            nn.ReLU(),
            # nn.MaxPool2d((2,2), (2,2)),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            # nn.MaxPool2d((2,2), (2,2)),
            nn.Conv2d(128, 64, 3),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear((NUM_ROWS-2-2-2)*(NUM_COLS-2-2-2)*64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_ROWS*NUM_COLS)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class DQNBot_base():

    def __init__(self):
        self.model = None
        self.memory = []
        self.memory_size = 2000
        self.won = 0

        # exploration
        self.epsilon = 1  # Exploration rate
        self.epsilon_min = 0.1  # Minimal exploration rate (epsilon-greedy)
        self.epsilon_decay = 0.998

        # learning
        self.batch_size = 128
        self.model_name = "Bot"
        self.tau = 0.8

        self.criterion = None
        self.opt = None
        self.scheduler = None

    def create_model(self):
        device = (
            "cuda"
            if torch.cuda.is_available()
            # else "mps"
            # if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")

        self.model = NeuralNetwork().to(device)

        self.criterion = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.0000001)
        self.scheduler = StepLR(self.opt, step_size=10000, gamma=0.5)

    def train_model(self, number_of_games):
        for i in range(number_of_games):
            self.train_game()

        if len(self.memory) > self.batch_size:
            self.replay(self.batch_size)

        self.decrease_epsilon()

    def train_game(self):
        game = GameLogic()
        for i in range(ceil(NUM_COLS*NUM_ROWS/3)):
            board = deepcopy(game.trainable_game())
            move = self.play(game)
            next_board, reward, done = game.rl_move(move)
            self.remember(board, move, reward, deepcopy(next_board), done)
            if done:
                self.won += 1
                break

    def play(self, game: GameLogic) -> int:
        # choose with probability epsilon a random move
        # balancing exploration and exploitation
        if random.random() < self.epsilon:
            move = random.randint(0, NUM_ROWS*NUM_COLS-1)
        else:
            with torch.no_grad():
                move = int(torch.argmax(self.model(torch.from_numpy(game.trainable_game()).float().unsqueeze(0))))
        return move

    # same as play but with very smallest epsilon
    def play_test(self, game: GameLogic) -> int:
        # choose with probability epsilon a random move
        # balancing exploration and exploitation
        if random.random() < self.epsilon_min:
            move = random.randint(0,  NUM_ROWS*NUM_COLS-1)
        else:
            with torch.no_grad():
                move = int(torch.argmax(self.model(torch.from_numpy(game.trainable_game()).float().unsqueeze(0))))
        return move

    def play_test_text(self, game):
        # choose with probability epsilon a random move
        # balancing exploration and exploitation
        if random.random() < self.epsilon_min:
            move = random.randint(0,  NUM_ROWS*NUM_COLS-1)
        else:
            with torch.no_grad():
                move = int(torch.argmax(self.model(torch.from_numpy(game.trainable_game()).float().unsqueeze(0))))
        with torch.no_grad():
            values = self.model(torch.from_numpy(game.trainable_game()).float().unsqueeze(0)).numpy()
        return move, values

    def remember(self, board, move, reward, next_board, done):
        self.memory.append((board, move, reward, next_board, done))
        while len(self.memory) > self.memory_size:
            rand_index = random.randint(0, len(self.memory) - 1)
            self.memory.pop(rand_index)

    def replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        for i in range(1):
            pred_batch = []
            target_batch = []
            game = GameLogic()

            for board, move, reward, next_board, done in batch:
                pred_reward = self.model(torch.from_numpy(board).float().unsqueeze(0))
                target_reward = torch.from_numpy(game.reward_all(board)).float().unsqueeze(0)

                pred_batch.append(pred_reward)
                target_batch.append(target_reward)

            pred_batch = torch.cat(pred_batch)
            target_batch = torch.cat(target_batch)

            self.opt.zero_grad()
            loss = self.criterion(pred_batch, target_batch)
            loss.backward()
            self.opt.step()
        # self.scheduler.step()
        # print(self.scheduler.get_last_lr())

    def decrease_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict()
        }, self.model_name + ".pth")

    def load(self):
        self.create_model()

        checkpoint = torch.load(self.model_name + '.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])

        self.model.train()

'''

# agent = DQNBot()
# agent.model_name = "TestModel"
# agent.create_model()
# game = GameLogic()
# game.move(2)
# print(agent.model(torch.from_numpy(game.trainable_game()).float().unsqueeze(0)))
# print(agent.model(torch.from_numpy(np.array([game.trainable_game(), game.trainable_game()])).float()))