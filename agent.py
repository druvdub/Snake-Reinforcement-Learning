import torch
import random
import numpy as np
from collections import deque

from game import SnakeAI, Direction, Point
from model import LinearQNet, QTrainer
from helper import plot

MAX_MEM = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self) -> None:
        self.no_games = 0  # number of games played
        self.epsilon = 0  # randomness
        self.gamma = 0.8  # discount rate

        # if maxlen is reached, oldest elements are removed
        self.memory = deque(maxlen=MAX_MEM)

        # 11 inputs, 512 neurons, 3 outputs
        self.model = LinearQNet(11, 512, 3)
        self.train = QTrainer(self.model, LR, self.gamma)  # LR = learning rate

    def get_state(self, game: SnakeAI) -> np.ndarray:
        head = game.snake[0]

        # +- 20 pixels because block size is 20
        up_point = Point(head.x, head.y - 20)
        down_point = Point(head.x, head.y + 20)
        left_point = Point(head.x - 20, head.y)
        right_point = Point(head.x + 20, head.y)

        up = game.direction == Direction.UP
        down = game.direction == Direction.DOWN
        left = game.direction == Direction.LEFT
        right = game.direction == Direction.RIGHT

        states = [
            # Collision up
            up and game.is_collision(up_point) or
            left and game.is_collision(left_point) or
            right and game.is_collision(right_point) or
            down and game.is_collision(down_point),

            # Collision right
            up and game.is_collision(right_point) or
            down and game.is_collision(left_point) or
            left and game.is_collision(up_point) or
            right and game.is_collision(down_point),

            # Collision left
            up and game.is_collision(left_point) or
            down and game.is_collision(right_point) or
            right and game.is_collision(up_point) or
            left and game.is_collision(down_point),

            # Directions
            up, down, left, right,

            # Food location
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right

        ]

        return np.array(states, dtype=int)

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, status: bool) -> None:
        self.memory.append((state, action, reward, next_state, status))

    def get_action(self, state: np.ndarray) -> int:
        # trade off exploration / exploitation
        self.epsilon = 120 - self.no_games
        fmove = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move_index = random.randint(0, 2)
            fmove[move_index] = 1
        else:
            _state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(_state)
            move = torch.argmax(prediction).item()
            fmove[move] = 1

        return fmove

    def train_long_memory(self) -> None:
        if BATCH_SIZE < len(self.memory):
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        # unpack
        states, actions, rewards, next_states, status = zip(*sample)
        self.train.train_step(states, actions, rewards, next_states, status)

    def train_short_memory(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, status: bool) -> None:
        self.train.train_step(state, action, reward, next_state, status)


def train():
    plot_scores = plot_mean_scores = list()

    total = record = 0

    agent = Agent()
    game = SnakeAI()

    while 1:
        # get prev state
        prev_state = agent.get_state(game)

        # fetch move
        move = agent.get_action(prev_state)

        # execute move and get new state
        reward, status, score = game.play_step(move)
        next_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(prev_state, move, reward, next_state, status)

        # remember
        agent.remember(prev_state, move, reward, next_state, status)

        if status:
            # train long memory
            game.reset()

            agent.no_games += 1
            agent.train_long_memory()

            if score > record:
                record = score

            print(
                f"Game: {agent.no_games} | Score: {score} | Record: {record}"
            )

            # plot
            plot_scores.append(score)
            total += score
            mean_score = total / agent.no_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
