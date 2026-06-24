TTT Env:
1. Reward high negative for illegal is bad for convergence
2. Action masking instead of illegal reward
3. Opponent in env break markov assumption and also not per standards to modern 2 player training pipelines


UTTT:
1. credit dilution
2. 


---

## Game Interface Design Decisions (src/games/)

These notes cover non-obvious choices in the AlphaZero game interface layer.
Reference: suragnair/alpha-zero-general (the canonical Python framework), AlphaZero.jl.

---

### Draw returns 1e-4, not 0.0 or 0

In Python, `0.0 == 0` is `True`. The standard MCTS termination check is:

```python
r = game.get_game_ended(canonical_board, player)
if r != 0:
    # terminal — backpropagate reward
```

Returning `0.0` for draw makes it indistinguishable from `0` (ongoing) under
this check. A board full of pieces with no winner would never terminate. The
suragnair reference uses `1e-4` specifically to avoid this: small enough to be
treated as near-zero reward, but non-zero so the `if r != 0` gate fires.

Ongoing returns int `0` (not float). The distinction matters if callers ever
do `isinstance(result, float)` to branch between draw and ongoing.

---

### get_next_state returns (next_state, -player)

suragnair MCTS unpacks the return value:

```python
next_s, next_player = game.get_next_state(canonical_board, 1, a)
```

Since TTT and UTTT are strictly alternating two-player games, `next_player`
is always `-player`. We return it anyway to keep the interface compatible with
any MCTS or self-play loop written against the suragnair contract, and to make
the alternation explicit at call sites rather than hardcoded in the MCTS.

---

### get_valid_moves accepts a player arg (but ignores it)

suragnair signature: `getValidMoves(board, player)`. In practice it is always
called on the canonical board (player=1), so the argument is never used — the
reference implementations for TTT and Connect4 both ignore it too. We accept
it with a default of 1 for drop-in compatibility. Valid moves depend only on
board occupancy, not on whose turn it is.

---

### get_string_representation returns bytes via .tobytes()

MCTS caches every visited node in a dict keyed by board state:

```python
s = game.get_string_representation(canonical_board)
self.Ns[s]        # visit count
self.Ps[s]        # policy prior
self.Qsa[(s, a)]  # Q-value
```

Without this method no MCTS implementation can store or look up nodes.
AlphaZero.jl formalises this as a hard constraint: states must be hashable.

For UTTT the key concatenates both `board.tobytes()` and `active.tobytes()`.
Both matter: two states with identical boards but different active masks are
genuinely different search nodes (the set of legal moves differs).

---

### All methods are @staticmethod — no instance state

suragnair uses instance methods with game config stored on `self` (e.g. board
size). Our design is fully stateless: state is always passed in and returned,
never stored. This is cleaner for functional self-play loops and matches the
approach taken by AlphaZero.jl. The tradeoff is that configurable variants
(e.g. 4x4 TTT) would need a new class rather than a constructor argument.

---

### get_board_winner uses equality check, not sum

For local boards (values 1, -1, 0) a sum check works fine. For the UTTT
meta_board (values 1, -1, 0, 2) a sum check produces false positives:
positions with values (1, 2, 0) sum to 3, which would be mistaken for a
three-in-a-row win. The equality check `board[a] == board[b] == board[c] == 1`
is immune to this.

---

### Methods not yet implemented (needed later for NN construction)

- `get_action_size() → int`: returns 9 (TTT) or 81 (UTTT). Used when building
  the policy head of the neural network.
- `get_board_size() → tuple`: the spatial shape of the NN input tensor.
  Not needed until the model architecture is defined.
