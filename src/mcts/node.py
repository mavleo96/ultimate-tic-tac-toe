from dataclasses import dataclass


@dataclass(slots=True)
class Node:
    """
    A node in the search tree. Statistics live on the child and describe the
    edge from its parent: `visits`/`value_sum` count and sum the values backed
    up through that edge, so Q = value_sum / visits is exactly what the parent
    needs when scoring the move that leads here.

    `value_sum` is accumulated from the parent-mover's perspective (backup
    negates once at the leaf), so Q feeds the parent's PUCT directly.
    """

    # Canonical form (mover always +1). None for stubs; filled when first selected.
    state: object = None
    # Prior for this move, from the parent's network evaluation.
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    # action -> Node, created at expansion. None means unexpanded (and stays
    # None for terminal nodes, which are never expanded).
    children: dict | None = None

    @property
    def q(self) -> float:
        # Mean value of this edge, computed on demand. Callers guard visits == 0.
        return self.value_sum / self.visits


@dataclass
class Tree:
    """
    One tree per game, alive for the whole game. Pure data — no in-flight search
    state is ever stored here (the runner holds the leaf/path between the two
    halves of a simulation).
    """

    root: Node

    def advance(self, action: int) -> None:
        # Play a move: the chosen child becomes the new root, siblings dropped.
        # Reuses the searched subtree so its statistics carry into the next move.
        self.root = self.root.children[action]
