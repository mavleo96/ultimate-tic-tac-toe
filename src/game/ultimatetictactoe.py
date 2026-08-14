from typing import NamedTuple

import numpy as np


class UTTTState(NamedTuple):
    board: np.ndarray  # (9, 9) int8
    active: np.ndarray  # (9,) bool


class UltimateTicTacToe:
    SYMM_AUG_INDICES_3x3 = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],  # identity
        [2, 5, 8, 1, 4, 7, 0, 3, 6],  # rot 90 CCW
        [8, 7, 6, 5, 4, 3, 2, 1, 0],  # rot 180
        [6, 3, 0, 7, 4, 1, 8, 5, 2],  # rot 270 CCW
        [2, 1, 0, 5, 4, 3, 8, 7, 6],  # flip horizontal
        [6, 7, 8, 3, 4, 5, 0, 1, 2],  # flip vertical
        [0, 3, 6, 1, 4, 7, 2, 5, 8],  # transpose (main diag)
        [8, 5, 2, 7, 4, 1, 6, 3, 0],  # anti-transpose
    ]

    # Each 9x9 symmetry is the same spatial transform applied independently to
    # "which sub-board" and "which cell within it": idx9[b*9+c] == idx3[b]*9 + idx3[c].
    SYMM_AUG_INDICES_9x9 = [
        [idx3[b] * 9 + idx3[c] for b in range(9) for c in range(9)] for idx3 in SYMM_AUG_INDICES_3x3
    ]

    WIN_LINES = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),  # rows
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),  # cols
        (0, 4, 8),
        (2, 4, 6),  # diags
    )

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
            return 1.0
        if winner == -player:
            return -1.0
        if winner == 2:
            # 1e-4 not 0.0: in Python 0.0 == 0 is True, so returning 0.0 for draw
            # makes it indistinguishable from ongoing (0) under any `if r != 0` check.
            return 1e-4
        return 0

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
