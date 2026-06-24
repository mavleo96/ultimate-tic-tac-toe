import logging

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

REWARD_MAP = {
    "win": 1.0,
    "loss": -1.0,
    "draw": 0.5,
    "illegal": -10.0,
}


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, opponent, render_mode=None) -> None:
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        super().__init__()

        self.render_mode = render_mode

        # Note: 9 shaped grid with opponent -1, empty 0 and self 1
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(-1, 1, shape=(9,), dtype=np.int8)

        self.agent_first = None
        self.board = None
        self.opponent = opponent

    def update_opponent(self, opponent):
        self.opponent = opponent

    def opponent_play(self, state):
        op_action = self.opponent.act(state)

        # Fallback: Random move
        if not self.action_space.contains(op_action) or state[op_action] != 0:
            logger.warning(
                f"Opponent provided invalid action {op_action}; Falling back to random move."
            )

            empty_cells = np.where(state == 0)[0]
            op_action = int(self.np_random.choice(empty_cells))

        return op_action

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.board = np.zeros(9, dtype=np.int8)

        self.agent_first = True
        if self.np_random.random() < 0.5:
            self.agent_first = False
            op_action = self.opponent_play(-self.board)
            self.board[op_action] = -1

        return self.board.copy(), {}

    def step(self, action):
        """
        Full-turn step: agent moves, then opponent responds.
        Board is always from the agent's perspective (agent=1, opponent=-1).
        Opponent receives the flipped board so it also sees itself as 1.

        info["result"]: "win" | "draw" | "loss" | "illegal" | None
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in [0, 8].")

        # Note: we terminate on illegal move
        if self.board[action] != 0:
            return self.board.copy(), REWARD_MAP["illegal"], True, False, {"result": "illegal"}

        self.board[action] = 1

        if self._check_winner(self.board, 1):
            return self.board.copy(), REWARD_MAP["win"], True, False, {"result": "win"}

        if self._is_board_full(self.board):
            return self.board.copy(), REWARD_MAP["draw"], True, False, {"result": "draw"}

        # flip the board entries so opponent agent see itself as 1
        op_action = self.opponent_play(-self.board)
        self.board[op_action] = -1

        if self._check_winner(self.board, -1):
            return self.board.copy(), REWARD_MAP["loss"], True, False, {"result": "loss"}

        if self._is_board_full(self.board):
            return self.board.copy(), REWARD_MAP["draw"], True, False, {"result": "draw"}

        return self.board.copy(), 0.0, False, False, {"result": None}

    @staticmethod
    def _check_winner(board, player):
        wins = (
            (0, 1, 2),  # row 0
            (3, 4, 5),  # row 1
            (6, 7, 8),  # row 2
            (0, 3, 6),  # col 0
            (1, 4, 7),  # col 1
            (2, 5, 8),  # col 2
            (0, 4, 8),  # diag 1
            (2, 4, 6),  # diag 2
        )

        for a, b, c in wins:
            s = board[a] + board[b] + board[c]
            if s == 3 * player:
                return True
        return False

    @staticmethod
    def _is_board_full(board):
        return not np.any(board == 0)

    def render(self):
        if self.render_mode == "ansi":
            symbols = {
                1: "X" if self.agent_first else "O",
                -1: "O" if self.agent_first else "X",
                0: ".",
            }
            board = [symbols[x] for x in self.board]
            return (
                f"{board[0]} {board[1]} {board[2]}\n"
                f"{board[3]} {board[4]} {board[5]}\n"
                f"{board[6]} {board[7]} {board[8]}"
            )


if __name__ == "__main__":
    from src.rl.opponent import RandomTTTOpponent

    op = RandomTTTOpponent()
    game = TicTacToeEnv(op, render_mode="ansi")

    help_str = {
        "idle": "press 's' to start a new game, 'q' to quit",
        "playing": "enter a move (0-8) or 'q' to quit",
    }

    started = False
    print(help_str["idle"])
    while True:
        cmd = input("enter cmd >> ").lower()
        if cmd == "q":
            break
        elif cmd == "s":
            _ = game.reset()
            print(game.render())
            started = True
        elif started and cmd in "012345678":
            _, _, terminated, _, status = game.step(int(cmd))
            print(game.render())
            if terminated:
                started = False
                print(f"game over: {status['result']}")
                print(help_str["idle"])
        else:
            print(help_str["idle"] if not started else help_str["playing"])
