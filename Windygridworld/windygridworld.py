import numpy as np
import os
import time


class WindyGridWorld:
    def __init__(self):
        self.width = 10
        self.height = 7
        self.grid = np.array(['O'] * 70).reshape([self.height, self.width])
        self.weak_wind = [3, 4, 5, 8]
        self.strong_wind = [6, 7]
        self.action_space = 4
        self.observation_space = 70
        self.action = {0: 'left', 1: 'up', 2: 'right', 3: 'down'}
        self.goal = [3, 7]

    def reset(self):
        self.state = [3, 0]
        self.grid = np.array(['O'] * 70).reshape([self.height, self.width])
        self.grid[self.state[0], self.state[1]] = 'X'
        return self.state

    def render(self, ):
        time.sleep(0.1)
        os.system('cls')
        print(self.grid)

    def step(self, action):
        # original action
        if action == 0:
            if self.state[1] != 0:
                self.state[1] -= 1
        elif action == 1:
            if self.state[0] != 0:
                self.state[0] -= 1
        elif action == 2:
            if self.state[1] != self.width - 1:
                self.state[1] += 1
        elif action == 3:
            if self.state[0] != self.height - 1:
                self.state[0] += 1

        else:
            print('올바르지 않은 action입니다.')

        # windy action
        if self.state[1] in self.weak_wind + self.strong_wind:
            if self.state[1] in self.weak_wind:
                if self.state[0] != 0:
                    self.state[0] -= 1
            else:
                if self.state[0] >= 2:
                    self.state[0] -= 2
                elif self.state[0] == 1:
                    self.state[0] -= 1

        self.grid = np.array(['O'] * 70).reshape([self.height, self.width])
        self.grid[self.state[0], self.state[1]] = 'X'

        if self.state == self.goal:
            return self.state, 0, True, None
        else:
            return self.state, -1, False, None
