
from .util import check_win

import numpy as np


class GameEnv:
    def __init__(self, board_size, win_mark):

        self.board_size = board_size
        
        # win_mark : 5 -> omok
        # win_mark : 3 -> tictaetoe
        self.win_mark = win_mark

        # No stone: 0, Black stone: 1, White stone = -1
        self.board = np.zeros([self.board_size, self.board_size])

        # black turn: 0, white turn: 1
        self.turn = 0

    def reset(self):
        # No stone: 0, Black stone: 1, White stone = -1
        self.board = np.zeros([self.board_size, self.board_size])
        # black turn: 0, white turn: 1
        self.turn = 0
        return self.board

    def step(self, action):

        if np.any(action) != 0:
            action_index = np.argmax(action)

            x_index = action_index // self.board_size
            y_index = action_index % self.board_size

        else:
            raise ValueError("Action is none.")

        if self.board[x_index, y_index] != 0:
            raise ValueError("No legal action.")

        # update board
        if self.turn == 0:
            self.board[x_index, y_index] = 1
            self.turn = 1

        else:
            self.board[x_index, y_index] = -1
            self.turn = 0

        # Check_win 0: playing, 1: black win, 2: white win, 3: draw
        win_index = check_win(self.board, self.win_mark)

        return self.board, win_index, self.turn

    @staticmethod
    def _check_win(game_board, win_mark):
        num_mark = np.count_nonzero(game_board)
        state_size = len(game_board)

        # check win
        for row in range(state_size - win_mark + 1):
            for col in range(state_size - win_mark + 1):
                current_grid = game_board[row: row + win_mark, col: col + win_mark]

                sum_horizontal = np.sum(current_grid, axis=1)
                sum_vertical = np.sum(current_grid, axis=0)
                sum_diagonal_1 = np.sum(current_grid.diagonal())
                sum_diagonal_2 = np.sum(np.flipud(current_grid).diagonal())

                # Black wins! (Horizontal and Vertical)
                if win_mark in sum_horizontal or win_mark in sum_vertical:
                    return 1

                # Black wins! (Diagonal)
                if win_mark == sum_diagonal_1 or win_mark == sum_diagonal_2:
                    return 1

                # White wins! (Horizontal and Vertical)
                if -win_mark in sum_horizontal or -win_mark in sum_vertical:
                    return 2

                # White wins! (Diagonal)
                if -win_mark == sum_diagonal_1 or -win_mark == sum_diagonal_2:
                    return 2

        # Draw (board is full)
        if num_mark == state_size * state_size:
            return 3

        # If No winner or no draw
        return 0
