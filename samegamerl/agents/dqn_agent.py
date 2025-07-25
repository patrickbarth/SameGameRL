import random
from copy import deepcopy
from math import ceil

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from samegamerl.agents.base_agent import BaseAgent
from samegamerl.game.game import Game
from samegamerl.game.game_params import NUM_COLORS, NUM_ROWS, NUM_COLS
from samegamerl.agents.replay_buffer import ReplayBuffer


"""
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
"""


class DqnAgent(BaseAgent):

    def __init__(
        self,
        model: nn.Module,
        model_name: str,  # used for saving the model
        learning_rate: float,
        initial_epsilon: float,  # determines how random the bot chooses it's actions
        epsilon_decay: float,  # how quickly the bot moves from exploration to exploitation
        final_epsilon: float,  # minimum rate of exploration
        batch_size: int = 128,  # number of game moves used for each training episode
        gamma: float = 0.95,  # how much future expected rewards count towards an action
        tau: float = 0.5,  # how quickly the target model adapts to the model
    ):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using {self.device} device")

        self.model = model.to(self.device)
        self.target_model = model.to(self.device)

        self.replay_buffer = ReplayBuffer(capacity=5000)
        self.batch_size = batch_size
        self.won = 0

        # exploration
        self.epsilon = initial_epsilon  # Exploration rate
        self.epsilon_min = final_epsilon  # Minimal exploration rate
        self.epsilon_decay = epsilon_decay

        # learning
        self.batch_size = batch_size
        self.gamma = gamma
        self.model_name = model_name
        self.tau = tau
        self.learning_rate = learning_rate

        self.criterion = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def act(self, observation: np.ndarray) -> int:
        # choose with probability epsilon a random move
        # balancing exploration and exploitation
        if random.random() < self.epsilon:
            move = random.randint(0, NUM_ROWS * NUM_COLS - 1)
        else:
            obs_tensor = (
                torch.tensor(observation, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                q_values = self.model(obs_tensor)
            move = q_values.argmax().item()
        return move

    def act_visualize(self, observation: np.ndarray) -> tuple[int, np.ndarray]:
        # choose with probability epsilon a random move
        # balancing exploration and exploitation
        if random.random() < self.epsilon:
            move = random.randint(0, NUM_ROWS * NUM_COLS - 1)
        else:
            obs_tensor = (
                torch.tensor(observation, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                q_values = self.model(obs_tensor)
            move = q_values.argmax().item()
        return move, q_values

    def remember(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def learn(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0

        states = self.replay_buffer.sample(self.batch_size)
        obs, actions, rewards, next_obs, dones = states

        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = (
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1).to(self.device)

        pred_q_values = self.model(obs).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_model(next_obs).max(1).values.unsqueeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (~dones)

        loss = self.criterion(pred_q_values, target_q_values)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss

    def decrease_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def update_target_model(self):
        target_model_state_dict = self.target_model.state_dict()
        model_state_dict = self.model.state_dict()
        for key in model_state_dict:
            target_model_state_dict[key] = model_state_dict[
                key
            ] * self.tau + target_model_state_dict[key] * (1 - self.tau)
        self.target_model.load_state_dict(target_model_state_dict)

    def save(self, name: str | None = None):
        if not name:
            name = self.model_name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
            },
            "samegamerl/models/" + name + ".pth",
        )

    def load(self, load_target=False, name: str | None = None):
        if not name:
            name = self.model_name
        checkpoint = torch.load("samegamerl/models/" + name + ".pth")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_target:
            self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        else:
            self.target_model.load_state_dict(checkpoint["model_state_dict"])

        self.model.train()
        self.target_model.eval()


"""
    def reset_model(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            # else "mps"
            # if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")

        self.model = NeuralNetwork().to(self.device)
        self.target_model = NeuralNetwork().to(self.device)

        self.criterion
        self.opt

        # self.criterion = torch.nn.MSELoss()
        # self.opt = torch.optim.Adam(self.model.parameters(), lr=0.000001)
        # self.scheduler = StepLR(self.opt, step_size=10000, gamma=0.5)
"""


"""
    def train_model(self, number_of_games):
        for i in range(number_of_games):
            self.train_game()

        if len(self.memory) > self.batch_size:
            self.replay(self.batch_size)

        self.update_target_model()
        self.decrease_epsilon()

    def train_game(self):
        game = Game()
        for i in range(ceil(NUM_COLS*NUM_ROWS/3)):
            board = deepcopy(game.trainable_game())
            move = self.play(game)
            next_board, reward, done = game.rl_move(move)
            self.remember(board, move, reward, deepcopy(next_board), done)
            if done:
                self.won += 1
                break

    def play(self, game: Game) -> int:
        # choose with probability epsilon a random move
        # balancing exploration and exploitation
        if random.random() < self.epsilon:
            move = random.randint(0, NUM_ROWS*NUM_COLS-1)
        else:
            with torch.no_grad():
                move = int(torch.argmax(self.model(torch.from_numpy(game.trainable_game()).float().unsqueeze(0))))
        return move

    # same as play but with very smallest epsilon
    def play_test(self, game: Game) -> int:
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

            for board, move, reward, next_board, done in batch:
                pred_reward = self.model(torch.from_numpy(board).float().unsqueeze(0))
                target_reward = pred_reward.clone().detach()

                with torch.no_grad():
                    if not done:
                        target = (reward + self.gamma * torch.max(self.target_model(torch.from_numpy(next_board).float().unsqueeze(0))))
                    else:
                        target = reward
                    target_reward[0][move] = target

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



# agent = DQNBot()
# agent.model_name = "TestModel"
# agent.create_model()
# game = GameLogic()
# game.move(2)
# print(agent.model(torch.from_numpy(game.trainable_game()).float().unsqueeze(0)))
# print(agent.model(torch.from_numpy(np.array([game.trainable_game(), game.trainable_game()])).float()))
"""
