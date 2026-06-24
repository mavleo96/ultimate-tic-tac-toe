import numpy as np


class TicTacToe:
    # 8 symmetries: 4 rotations x 2 flips
    SYMM_AUG_INDICES = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],  # identity
        [2, 5, 8, 1, 4, 7, 0, 3, 6],  # rot 90 CCW
        [8, 7, 6, 5, 4, 3, 2, 1, 0],  # rot 180
        [6, 3, 0, 7, 4, 1, 8, 5, 2],  # rot 270 CCW
        [2, 1, 0, 5, 4, 3, 8, 7, 6],  # flip horizontal
        [6, 7, 8, 3, 4, 5, 0, 1, 2],  # flip vertical
        [0, 3, 6, 1, 4, 7, 2, 5, 8],  # transpose (main diag)
        [8, 5, 2, 7, 4, 1, 6, 3, 0],  # anti-transpose
    ]

    WIN_LINES = (
        (0, 1, 2),  # row 0
        (3, 4, 5),  # row 1
        (6, 7, 8),  # row 2
        (0, 3, 6),  # col 0
        (1, 4, 7),  # col 1
        (2, 5, 8),  # col 2
        (0, 4, 8),  # diag 1
        (2, 4, 6),  # diag 2
    )

    @staticmethod
    def get_init_state():
        return np.zeros(9, dtype=np.int8)

    @staticmethod
    def get_next_state(state, player, action):
        assert state[action] == 0

        next_state = state.copy()
        next_state[action] = player
        return next_state, -player

    @staticmethod
    def get_valid_moves(state, player=1):
        return (state == 0).astype(np.int8)

    @staticmethod
    def get_game_ended(state, player):
        for a, b, c in TicTacToe.WIN_LINES:
            s = state[a] + state[b] + state[c]
            if s == 3 * player:
                return 1.0
            if s == -3 * player:
                return -1.0
        if not np.any(state == 0):
            # 1e-4 not 0.0: in Python 0.0 == 0 is True, so returning 0.0 for draw
            # makes it indistinguishable from ongoing (0) under any `if r != 0` check.
            return 1e-4
        return 0

    @staticmethod
    def get_canonical_form(state, player):
        return state * player

    @staticmethod
    def get_symmetries(state, pi):
        # pi = np.asarray(pi)
        return [(state[idx], pi[idx]) for idx in TicTacToe.SYMM_AUG_INDICES]

    @staticmethod
    def get_string_representation(state):
        return state.tobytes()
