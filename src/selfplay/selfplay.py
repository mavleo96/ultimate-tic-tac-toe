from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from ..mcts.node import Node, Tree
from ..mcts.searcher import Searcher
from ..model.wrapper import NetworkWrapper


@dataclass
class SelfPlayConfig:
    n_episodes: int = 100
    temp_threshold: int = 10
    n_concurrent: int = 64
    seed: int = 0


class SelfPlay:
    def __init__(
        self,
        game,
        network: NetworkWrapper,
        searcher: Searcher,
        config: SelfPlayConfig,
    ) -> None:
        self.game = game
        self.network = network
        self.searcher = searcher
        self.config = config

    def run(self) -> list[tuple]:
        all_examples = []
        n = self.config.n_episodes
        with tqdm(total=n, desc="self-play", unit="ep", leave=False) as bar:
            for start in range(0, n, self.config.n_concurrent):
                batch = range(start, min(start + self.config.n_concurrent, n))
                all_examples.extend(self._run_batch(batch))
                bar.update(len(batch))
        return all_examples

    def _run_batch(self, indices) -> list[tuple]:
        live = []
        for i in indices:
            gen = self._play_game(np.random.default_rng(self.config.seed + i))
            live.append((gen, next(gen)))

        examples = []
        while live:
            priors, values = self.network.predict([state for _, state in live])
            advanced = []
            for (gen, _), pi, value in zip(live, priors, values, strict=True):
                try:
                    advanced.append((gen, gen.send((pi, float(value)))))
                except StopIteration as finished:
                    examples.extend(finished.value)
            live = advanced
        return examples

    def _play_game(self, rng: np.random.Generator):
        """
        One self-play game as a generator. Yields the canonical leaf state it needs
        evaluated and is resumed with (priors, value); returns the game's training
        examples on completion.

        Sim/move counters and the pending leaf/path stay ordinary locals — that is
        the whole point of using a generator: the pause between the two halves of a
        simulation lives here, not on the (shared) searcher or the (pure) tree.

        Must never call self.network: evaluation belongs to the scheduler
        (_run_batch), which is what lets many games share one batched call.
        """
        state = self.game.get_init_state()
        player = 1
        n_actions = len(self.game.get_valid_moves(state, player))
        tree = Tree(Node(state=self.game.get_canonical_form(state, player)))
        records = []  # (canonical_root_state, pi float32, player_at_move)
        move = 0

        while True:
            terminated, value = self.game.get_game_ended(state, player)
            if terminated:
                break

            for _ in range(self.searcher.config.n_simulations):
                leaf, path = self.searcher.descend(tree, rng)
                leaf_terminated, leaf_value = self.game.get_game_ended(leaf.state, 1)
                if leaf_terminated:
                    # True value, no network call needed.
                    self.searcher.backup(leaf, path, None, leaf_value)
                else:
                    priors, leaf_value = yield leaf.state
                    self.searcher.backup(leaf, path, priors, leaf_value)

            pi = self._visit_distribution(tree.root, n_actions)
            records.append((tree.root.state, pi.astype(np.float32), player))

            temp = 1.0 if move < self.config.temp_threshold else 0.0
            action = self._sample_action(pi, temp, rng)
            state, player = self.game.get_next_state(state, player, action)
            tree.advance(action)
            move += 1

        # `value` is from `player`'s perspective at game end; flip per stored player.
        return [(s, pi, value if p == player else -value) for s, pi, p in records]

    @staticmethod
    def _visit_distribution(root: Node, n_actions: int) -> np.ndarray:
        counts = np.zeros(n_actions)
        for action, child in root.children.items():
            counts[action] = child.visits
        if counts.sum() == 0:
            counts[list(root.children)] = 1.0
        return counts / counts.sum()

    @staticmethod
    def _sample_action(pi: np.ndarray, temp: float, rng: np.random.Generator) -> int:
        # Temperature applies to visit counts, but pi is those counts normalized, so
        # sharpening pi is the same thing.
        if temp < 1e-8:
            # Break argmax ties randomly — lowest-index tie-breaking measurably skews
            # UTTT openings.
            return int(rng.choice(np.flatnonzero(pi == pi.max())))
        weights = pi ** (1.0 / temp)
        return int(rng.choice(len(pi), p=weights / weights.sum()))
