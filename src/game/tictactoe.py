from typing import NamedTuple

import numpy as np

from .utils import SYMM_INDICES_3x3, WIN_LINES_3x3


class TTTState(NamedTuple):
    """
    board:
    np.ndarray of shape (9,)

    Sign convention:
    1: self player
    -1: opponent player
    0: empty

    Example:
    [1, -1, 0, -1, -1, 0, 0, 1, 1]

    X | O | -
    O | O | -
    - | X | X
    """

    board: np.ndarray


class TicTacToe:
    SYMM_AUG_INDICES = SYMM_INDICES_3x3
    WIN_LINES = WIN_LINES_3x3

    @staticmethod
    def get_init_state():
        return TTTState(board=np.zeros(9, dtype=np.int8))

    @staticmethod
    def get_next_state(state, player, action):
        board = state.board
        assert board[action] == 0

        next_board = board.copy()
        next_board[action] = player
        return TTTState(board=next_board), -player

    @staticmethod
    def get_valid_moves(state, player=1):
        return (state.board == 0).astype(np.int8)

    @staticmethod
    def get_game_ended(state, player):
        board = state.board
        for a, b, c in TicTacToe.WIN_LINES:
            s = board[a] + board[b] + board[c]
            if s == 3 * player:
                return True, 1.0
            if s == -3 * player:
                return True, -1.0
        if not np.any(board == 0):
            return True, 0.0
        return False, 0.0

    @staticmethod
    def get_canonical_form(state, player):
        return TTTState(board=state.board * player)

    @staticmethod
    def get_symmetries(state, pi):
        board = state.board
        return [(TTTState(board=board[idx]), pi[idx]) for idx in TicTacToe.SYMM_AUG_INDICES]

    @staticmethod
    def get_string_representation(state):
        return state.board.tobytes()
