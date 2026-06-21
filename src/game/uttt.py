import gymnasium as gym
import numpy as np
from gymnasium import spaces

REWARD_MAP = {
    "local_win": 0.1,
    "local_loss": -0.1,
    "win": 1.0,
    "loss": -1.0,
    "draw": 0.5,
    "illegal": -10.0,
}


class UltimateTicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, opponent, render_mode=None) -> None:
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        super().__init__()

        self.render_mode = render_mode

        # Note:
        # 1. actions space is flattened space of 81; board is a // 9 and cell is a % 9
        # 2. observation space is dict of:
        #    a. current board (9 x 9)
        #    b. active boards (9,) - binary mask of boards where the next move can be played
        self.action_space = spaces.Discrete(81)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(-1, 1, shape=(9, 9), dtype=np.int8),
                "active": spaces.MultiBinary(9),
            }
        )

        self.agent_first = None
        self.state = {"board": None, "active": None}
        # Note: meta_board tracks the status of each local board: empty 0, self 1,
        #       opponent -1, draw 2; it is derived from state but we keep it for convenience
        self.meta_board = None
        self.opponent = opponent

    def update_opponent(self, opponent):
        self.opponent = opponent

    def opponent_play(self, state):
        op_action = self.opponent.act(state)

        # Fallback: Random move
        if not self.action_space.contains(op_action) or not self._validate_action(op_action):
            active_boards = np.where(state["active"])[0]
            board = int(self.np_random.choice(active_boards))
            empty_cells = np.where(state["board"][board] == 0)[0]
            cell = int(self.np_random.choice(empty_cells))
            op_action = board * 9 + cell

        return op_action

    @property
    def obs(self):
        return {
            "board": self.state["board"].copy(),
            "active": self.state["active"].copy(),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.state["board"] = np.zeros((9, 9), dtype=np.int8)
        self.state["active"] = np.ones(9, dtype=bool)
        self.meta_board = np.zeros(9, dtype=np.int8)

        self.agent_first = True
        if self.np_random.random() < 0.5:
            self.agent_first = False
            op_action = self.opponent_play(
                {"board": -self.state["board"], "active": self.state["active"]}
            )
            self._update_board(op_action, -1)

        return self.obs, {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in [0, 80].")

        reward = 0.0

        # Note: we terminate on illegal move
        if not self._validate_action(action):
            reward += REWARD_MAP["illegal"]
            return self.obs, reward, True, False, {"result": "illegal"}

        prev_local_status = self.meta_board[action // 9]
        self._update_board(action, 1)
        if prev_local_status == 0 and self.meta_board[action // 9] == 1:
            reward += REWARD_MAP["local_win"]

        if self._check_winner(1):
            reward += REWARD_MAP["win"]
            return self.obs, reward, True, False, {"result": "win"}

        if self._is_game_over():
            reward += REWARD_MAP["draw"]
            return self.obs, reward, True, False, {"result": "draw"}

        # flip the board entries so opponent agent see itself as 1
        op_action = self.opponent_play(
            {"board": -self.state["board"], "active": self.state["active"]}
        )

        prev_local_status = self.meta_board[op_action // 9]
        self._update_board(op_action, -1)
        if prev_local_status == 0 and self.meta_board[op_action // 9] == -1:
            reward += REWARD_MAP["local_loss"]

        if self._check_winner(-1):
            reward += REWARD_MAP["loss"]
            return self.obs, reward, True, False, {"result": "loss"}

        if self._is_game_over():
            reward += REWARD_MAP["draw"]
            return self.obs, reward, True, False, {"result": "draw"}

        return self.obs, reward, False, False, {"result": None}

    def _validate_action(self, action):
        if not self.state["active"][action // 9]:
            return False
        if self.state["board"][action // 9, action % 9] != 0:
            return False
        return True

    def _update_board(self, action, player):
        # 1. update board state
        self.state["board"][action // 9, action % 9] = player
        # 2. update meta board
        self.meta_board[action // 9] = self.get_board_winner(self.state["board"][action // 9])
        # 3. update active state for next turn
        if self.meta_board[action % 9] != 0:
            self.state["active"] = self.meta_board == 0
        else:
            self.state["active"] = np.zeros(9, dtype=bool)
            self.state["active"][action % 9] = True

    @staticmethod
    def get_board_winner(board):
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
            if board[a] == board[b] == board[c] == 1:
                return 1
            if board[a] == board[b] == board[c] == -1:
                return -1

        return 2 if all(x != 0 for x in board) else 0

    def _check_winner(self, player):
        return self.get_board_winner(self.meta_board) == player

    def _is_game_over(self):
        return self.get_board_winner(self.meta_board) == 2

    def render(self):
        if self.render_mode == "ansi":
            BIG_X = ["\\   /", "  X  ", "/   \\"]
            BIG_O = ["/---\\", "|   |", "\\---/"]
            BIG_DRAW = ["=====", "=====", "====="]
            DIVIDER = "------+-------+------"
            symbols = {
                1: "X" if self.agent_first else "O",
                -1: "O" if self.agent_first else "X",
                0: ".",
            }
            big_symbols = {
                1: BIG_X if self.agent_first else BIG_O,
                -1: BIG_O if self.agent_first else BIG_X,
                2: BIG_DRAW,
            }

            def single_board_string(board):
                board = [symbols[x] for x in board]
                strings = [
                    f"{board[0]} {board[1]} {board[2]}",
                    f"{board[3]} {board[4]} {board[5]}",
                    f"{board[6]} {board[7]} {board[8]}",
                ]
                return strings

            board_strings = [
                big_symbols.get(self.meta_board[i], single_board_string(self.state["board"][i]))
                for i in range(9)
            ]

            for i in range(9):
                substrings = []
                for j in range(3):
                    s = board_strings[i // 3 * 3 + j][i % 3]
                    substrings.append(s)
                line = " | ".join(substrings)
                print(line)
                if i % 3 == 2 and i != 8:
                    print(DIVIDER)

            active_board_list = np.where(self.state["active"])[0].tolist()
            print(f"active boards: {active_board_list}")


if __name__ == "__main__":
    import re

    from src.rl.opponent import RandomUTTTOpponent

    op = RandomUTTTOpponent()
    game = UltimateTicTacToeEnv(op, render_mode="ansi")

    help_str = {
        "idle": "press 's' to start a new game, 'q' to quit",
        "playing": "enter a move as board (0-8) space cell (0-8) or 'q' to quit",
    }

    started = False
    print(help_str["idle"])
    while True:
        cmd = input("enter cmd >> ").lower()
        if cmd == "q":
            break
        elif cmd == "s":
            _ = game.reset()
            game.render()
            started = True
        elif started and re.match(r"^\d \d$", cmd):
            b, c = cmd.split()
            state, _, terminated, _, status = game.step(int(b) * 9 + int(c))
            game.render()
            if terminated:
                started = False
                print(f"game over: {status['result']}")
                print(help_str["idle"])
        else:
            print(help_str["idle"] if not started else help_str["playing"])
