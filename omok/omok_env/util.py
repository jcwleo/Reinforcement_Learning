import numpy as np

ALPHABET = ' A B C D E F G H I J K L M N O P Q R S'


def legal_actions(game_board):
    actions = []
    count = 0
    state_size = len(game_board)

    for i in range(state_size):
        for j in range(state_size):
            if game_board[i][j] == 0:
                actions.append([(i, j), count])
            count += 1

    return actions


# Check win
def check_win(game_board, win_mark):
    num_mark = np.count_nonzero(game_board)
    state_size = len(game_board)

    current_grid = np.zeros([win_mark, win_mark])

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


def render_str(gameboard, GAMEBOARD_SIZE, action_index):
    if action_index is not None:
        row = action_index // GAMEBOARD_SIZE
        col = action_index % GAMEBOARD_SIZE
    count = np.count_nonzero(gameboard)
    board_str = '\n  {}\n'.format(ALPHABET[:GAMEBOARD_SIZE * 2])
    for i in range(GAMEBOARD_SIZE):
        for j in range(GAMEBOARD_SIZE):
            if j == 0:
                board_str += '{:2}'.format(i + 1)
            if gameboard[i][j] == 0:
                if count > 0:
                    if col + 1 < GAMEBOARD_SIZE:
                        if (i, j) == (row, col + 1):
                            board_str += '.'
                        else:
                            board_str += ' .'
                    else:
                        board_str += ' .'
                else:
                    board_str += ' .'
            if gameboard[i][j] == 1:
                if (i, j) == (row, col):
                    board_str += '(O)'
                elif (i, j) == (row, col + 1):
                    board_str += 'O'
                else:
                    board_str += ' O'
            if gameboard[i][j] == -1:
                if (i, j) == (row, col):
                    board_str += '(X)'
                elif (i, j) == (row, col + 1):
                    board_str += 'X'
                else:
                    board_str += ' X'
            if j == GAMEBOARD_SIZE - 1:
                board_str += ' \n'
        if i == GAMEBOARD_SIZE - 1:
            board_str += '  ' + '-' * (GAMEBOARD_SIZE - 6) + \
                '  MOVE: {:2}  '.format(count) + '-' * (GAMEBOARD_SIZE - 6)
    print(board_str)

def get_state_pt(id, turn, state_size, channel_size):
    state = np.zeros([channel_size, state_size, state_size], 'float')
    length_game = len(id)

    state_1 = np.zeros([state_size, state_size], 'float')
    state_2 = np.zeros([state_size, state_size], 'float')

    channel_idx = channel_size - 1

    for i in range(length_game):
        row_idx = int(id[i] / state_size)
        col_idx = int(id[i] % state_size)

        if i != 0:
            if i % 2 == 0:
                state_1[row_idx, col_idx] = 1
            else:
                state_2[row_idx, col_idx] = 1

        if length_game - i < channel_size:
            channel_idx = length_game - i - 1

            if i % 2 == 0:
                state[channel_idx] = state_1
            else:
                state[channel_idx] = state_2

    if turn == 0:
        state[channel_size - 1] = 0
    else:
        state[channel_size - 1] = 1

    return state


def get_action(pi, tau):
    action_size = len(pi)
    action = np.zeros(action_size)
    action_index = np.random.choice(action_size, p=pi)
    action[action_index] = 1
    return action, action_index
