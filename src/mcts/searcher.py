from dataclasses import dataclass

import numpy as np

from .node import Node


@dataclass
class SearcherConfig:
    n_simulations: int = 100
    c_puct: float = 1.0
    use_root_noise: bool = False
    dirichlet_epsilon: float = 0.25


class Searcher:
    """
    Stateless AlphaZero-style MCTS. Holds config only — no per-game state — so a
    single instance serves all concurrent games.

    A simulation is split into two non-recursive, non-blocking halves because the
    network call can't happen mid-descent:

        descend(tree, rng) -> (leaf, path)   walk to an unexpanded/terminal node
        backup(leaf, path, priors, value)    expand + propagate the value up

    The caller holds `leaf` and `path` between the halves and decides whether the
    leaf is terminal (true value, no network) or needs a network evaluation.

    Values use negamax: `backup` negates the value at each level (good for me is
    bad for you). Selection uses PUCT with first-play urgency for unvisited edges.

    Does not import torch: the network is reached only via the caller, which does
    the (batched) evaluation.
    """

    def __init__(self, game, config: SearcherConfig):
        self.game = game
        self.config = config

    def descend(self, tree, rng: np.random.Generator):
        # Walk from the root by PUCT until an unexpanded or terminal node (both
        # have children is None). Iterative; returns the leaf plus the path
        # traversed to reach it (root first), which backup walks in reverse.
        node = tree.root
        path = [node]
        while node.children is not None:
            action, child = self._select(node, node is tree.root, rng)
            if child.state is None:
                # Stub selected for the first time — realize its canonical state
                # from the parent (we always play as +1 on the canonical board).
                raw, _ = self.game.get_next_state(node.state, 1, action)
                child.state = self.game.get_canonical_form(raw, -1)
            path.append(child)
            node = child
        return node, path

    def backup(self, leaf: Node, path: list[Node], priors, value: float) -> None:
        # priors is None for terminal leaves (true value, no expansion); otherwise
        # expand the leaf with the network's priors before propagating.
        if priors is not None:
            self._expand(leaf, priors)

        # Negate first so the leaf's own edge is credited from its parent's
        # perspective, then alternate up the tree.
        for node in reversed(path):
            node.visits += 1
            value = -value
            node.value_sum += value

    def _expand(self, node: Node, priors: np.ndarray) -> None:
        valid = self.game.get_valid_moves(node.state, 1)
        pi = priors * valid
        pi_sum = pi.sum()
        if pi_sum > 0:
            pi = pi / pi_sum
        else:
            # All valid moves had zero prior — fall back to uniform.
            pi = valid.astype(np.float64)
            pi /= pi.sum()

        node.children = {a: Node(prior=float(pi[a])) for a in range(len(valid)) if valid[a]}

    def _select(self, node: Node, is_root: bool, rng: np.random.Generator):
        # None means "use each child's stored prior" — only the noisy root pays
        # for a per-descent priors dict.
        priors = self._noisy_priors(node, rng) if is_root and self.config.use_root_noise else None

        sqrt_n = np.sqrt(node.visits)
        # First-play urgency: seed unvisited edges from the mean Q of already-
        # visited siblings (0.0 when none), so exploration isn't biased toward
        # unvisited moves in clearly winning/losing positions.
        visited = [c.q for c in node.children.values() if c.visits > 0]
        fpu = sum(visited) / len(visited) if visited else 0.0

        best_score = -np.inf
        best = None
        for action, child in node.children.items():
            q = child.q if child.visits > 0 else fpu
            prior = child.prior if priors is None else priors[action]
            u = self.config.c_puct * prior * sqrt_n / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best = (action, child)
        return best

    def _noisy_priors(self, node: Node, rng: np.random.Generator) -> dict:
        # Root Dirichlet noise is applied at selection time, on every move,
        # rather than baked into stored priors at expansion — a reused (already
        # expanded) root would otherwise silently skip noise entirely. Drawn
        # fresh per descent, which keeps the searcher stateless and the tree pure.
        eps = self.config.dirichlet_epsilon
        actions = list(node.children)
        # alpha scales inversely with the action count so one epsilon works
        # across TTT (9 actions) and UTTT (81).
        alpha = 10.0 / len(actions)
        noise = rng.dirichlet([alpha] * len(actions))
        return {
            a: (1 - eps) * node.children[a].prior + eps * n
            for a, n in zip(actions, noise, strict=True)
        }
