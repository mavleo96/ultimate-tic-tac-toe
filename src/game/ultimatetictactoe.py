from typing import NamedTuple

import numpy as np

from .utils import SYMM_INDICES_3x3, WIN_LINES_3x3


class UTTTState(NamedTuple):
    """
    board:
    np.ndarray of shape (9, 9)
    int8

    Sign convention:
    1: self player
    -1: opponent player
    0: empty

    active:
    np.ndarray of shape (9,) bool
    True if the board is active, False if it is not.
    """

    board: np.ndarray  # (9, 9) int8
    active: np.ndarray  # (9,) bool


class UltimateTicTacToe:
    SYMM_AUG_INDICES_3x3 = SYMM_INDICES_3x3

    # Each 9x9 symmetry is the same spatial transform applied independently to
    # "which sub-board" and "which cell within it": idx9[b*9+c] == idx3[b]*9 + idx3[c].
    SYMM_AUG_INDICES_9x9 = [
        [idx3[b] * 9 + idx3[c] for b in range(9) for c in range(9)] for idx3 in SYMM_INDICES_3x3
    ]

    WIN_LINES = WIN_LINES_3x3

    @staticmethod
    def get_init_state():
        return UTTTState(
            board=np.zeros((9, 9), dtype=np.int8),
            active=np.ones(9, dtype=bool),
        )

    @staticmethod
    def get_next_state(state, player, action):
        b, c = action // 9, action % 9
        assert state.active[b], f"action {action} targets board {b} which is not active"
        assert state.board[b, c] == 0, (
            f"action {action} targets cell ({b},{c}) which is already occupied"
        )
        board = state.board.copy()
        board[b, c] = player
        meta_board = UltimateTicTacToe._derive_meta_board(board)
        # The cell played (c) determines which macro board the opponent must play in.
        if meta_board[c] != 0:
            # Target board is already decided — opponent may play in any open board.
            active = meta_board == 0
        else:
            active = np.zeros(9, dtype=bool)
            active[c] = True
        return UTTTState(board=board, active=active), -player

    @staticmethod
    def get_valid_moves(state, player=1):
        # Valid if the board is active and the cell is empty.
        valid = np.zeros(81, dtype=np.int8)
        for b in np.argwhere(state.active).reshape(-1):
            valid[b * 9 : (b + 1) * 9] = state.board[b] == 0
        return valid

    @staticmethod
    def get_game_ended(state, player):
        meta_board = UltimateTicTacToe._derive_meta_board(state.board)
        winner = UltimateTicTacToe._get_board_winner(meta_board)
        if winner == player:
            return True, 1.0
        if winner == -player:
            return True, -1.0
        if winner == 2:
            return True, 0.0
        return False, 0.0

    @staticmethod
    def get_canonical_form(state, player):
        return UTTTState(
            board=state.board.copy() * player,
            active=state.active.copy(),
        )

    @staticmethod
    def get_symmetries(state, pi):
        result = []
        for idx9, idx3 in zip(
            UltimateTicTacToe.SYMM_AUG_INDICES_9x9,
            UltimateTicTacToe.SYMM_AUG_INDICES_3x3,
            strict=True,
        ):
            new_board = state.board.reshape(-1)[idx9].reshape(9, 9)
            new_active = state.active[idx3]
            new_pi = pi[idx9]
            result.append((UTTTState(board=new_board, active=new_active), new_pi))
        return result

    @staticmethod
    def get_string_representation(state):
        return state.board.tobytes() + state.active.tobytes()

    @staticmethod
    def _get_board_winner(board):
        for a, b, c in UltimateTicTacToe.WIN_LINES:
            if board[a] == board[b] == board[c] == 1:
                return 1
            if board[a] == board[b] == board[c] == -1:
                return -1
        return 2 if all(x != 0 for x in board) else 0

    @staticmethod
    def _derive_meta_board(board):
        meta = np.empty(9, dtype=np.int8)
        for i in range(9):
            meta[i] = UltimateTicTacToe._get_board_winner(board[i])
        return meta
