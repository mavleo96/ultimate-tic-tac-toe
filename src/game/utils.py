# Geometry of a 3x3 board, shared by TicTacToe and by every level of
# UltimateTicTacToe (sub-boards and the meta-board are all 3x3).

WIN_LINES_3x3 = (
    (0, 1, 2),  # row 0
    (3, 4, 5),  # row 1
    (6, 7, 8),  # row 2
    (0, 3, 6),  # col 0
    (1, 4, 7),  # col 1
    (2, 5, 8),  # col 2
    (0, 4, 8),  # diag 1
    (2, 4, 6),  # diag 2
)

# 8 symmetries: 4 rotations x 2 flips
SYMM_INDICES_3x3 = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],  # identity
    [2, 5, 8, 1, 4, 7, 0, 3, 6],  # rot 90 CCW
    [8, 7, 6, 5, 4, 3, 2, 1, 0],  # rot 180
    [6, 3, 0, 7, 4, 1, 8, 5, 2],  # rot 270 CCW
    [2, 1, 0, 5, 4, 3, 8, 7, 6],  # flip horizontal
    [6, 7, 8, 3, 4, 5, 0, 1, 2],  # flip vertical
    [0, 3, 6, 1, 4, 7, 2, 5, 8],  # transpose (main diag)
    [8, 5, 2, 7, 4, 1, 6, 3, 0],  # anti-transpose
]
