from dataclasses import dataclass

import numpy as np


@dataclass
class MCTSConfig:
    n_simulations: int = 100
    c_puct: float = 1.0
    temp_threshold: int = 10


class MCTS:
    def __init__(self, game, network, config: MCTSConfig):
        self.game = game
        self.network = network
        self.config = config

        self.Ps = {}  # s -> prior probs (n_actions,)
        self.Vs = {}  # s -> valid moves mask
        self.Ns = {}  # s -> visit count
        self.Qs = {}  # (s, a) -> mean action value
        self.Nsa = {}  # (s, a) -> visit count

    def get_action_probs(self, state, player: int, temp: float = 1.0) -> np.ndarray:
        for _ in range(self.config.n_simulations):
            self._search(state, player)

        canonical = self.game.get_canonical_form(state, player)
        s = self.game.get_string_representation(canonical)
        n_actions = len(self.Ps[s])

        counts = np.array([self.Nsa.get((s, a), 0) for a in range(n_actions)], dtype=np.float64)

        if counts.sum() == 0:
            # Root was expanded but no child was ever selected (n_simulations=1).
            # Fall back to valid-move uniform.
            probs = self.Vs[s].astype(np.float64)
            return probs / probs.sum()

        if temp < 1e-8:
            probs = np.zeros(n_actions)
            probs[np.argmax(counts)] = 1.0
            return probs

        counts = counts ** (1.0 / temp)
        return counts / counts.sum()

    def _search(self, state, player: int) -> float:
        canonical = self.game.get_canonical_form(state, player)
        s = self.game.get_string_representation(canonical)

        result = self.game.get_game_ended(canonical, 1)
        if result != 0:
            return -result

        if s not in self.Ps:
            # Leaf node — expand.
            pi, v = self.network.predict(canonical)
            valid = self.game.get_valid_moves(canonical, 1)
            pi = pi * valid
            pi_sum = pi.sum()
            if pi_sum > 0:
                pi /= pi_sum
            else:
                # All valid moves had zero prior — fall back to uniform.
                pi = valid.astype(np.float64)
                pi /= pi.sum()
            self.Ps[s] = pi
            self.Vs[s] = valid
            self.Ns[s] = 0
            return -v

        # Select action via UCB.
        valid = self.Vs[s]
        sqrt_ns = np.sqrt(self.Ns[s])
        best_score = -np.inf
        best_action = -1
        for a in range(len(valid)):
            if not valid[a]:
                continue
            q = self.Qs.get((s, a), 0.0)
            u = self.config.c_puct * self.Ps[s][a] * sqrt_ns / (1 + self.Nsa.get((s, a), 0))
            score = q + u
            if score > best_score:
                best_score = score
                best_action = a

        next_state, _ = self.game.get_next_state(canonical, 1, best_action)
        v = self._search(next_state, -1)

        sa = (s, best_action)
        n = self.Nsa.get(sa, 0)
        self.Qs[sa] = (n * self.Qs.get(sa, 0.0) + v) / (n + 1)
        self.Nsa[sa] = n + 1
        self.Ns[s] += 1
        return -v
