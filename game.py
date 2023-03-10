import pygame
import random
from enum import Enum
from collections import namedtuple  # tuple with named fields
import numpy as np

pygame.init()  # initialize pygame
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')


WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (4, 201, 116)
DGREEN = (3, 82, 48)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40


class SnakeAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode(
            (self.w, self.h))  # set display size
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()  # set game speed

        self.reset()  # init game state

    def reset(self):
        # init state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [
            self.head,
            Point(self.head.x-BLOCK_SIZE, self.head.y),
            Point(self.head.x-(2*BLOCK_SIZE), self.head.y)
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        # place food on the screen randomly
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE

        self.food = Point(x, y)

        # check if food is on the snake
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # move
        self._move(action)

        self.snake.insert(0, self.head)

        # check if game over
        reward = 0
        game_over = False

        # check if game over
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            reward = -10
            game_over = True
            return reward, game_over, self.score

        # place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def is_collision(self, pt=None):

        if pt is None:
            pt = self.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):

        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(
                pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

            pygame.draw.rect(self.display, DGREEN,
                             pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

        pygame.display.flip()

    def _move(self, action):

        # action[up, right, left]

        clock_wise = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP
        ]

        # convert to clockwise
        idx = clock_wise.index(self.direction)

        # get new direction
        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[idx]

        # right turn
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_direction = clock_wise[next_idx]

        # left turn
        else:
            next_idx = (idx - 1) % 4
            new_direction = clock_wise[next_idx]

        self.direction = new_direction

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
