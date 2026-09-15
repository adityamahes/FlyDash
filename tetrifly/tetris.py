"""The simplest Tetris that still makes sense as a control problem.

Deliberately left out, because the agent has only four controls and the first
iteration should be easy to reason about: no next-piece preview, no hold slot,
no soft or hard drop, no wall kicks, no levels or speed-up, no multi-line score
bonus. Gravity pulls the piece down exactly one row per frame, so a frame is a
fixed unit of time for both the game and the brain simulation.

Coordinates are (row, column) with row 0 at the top of the board.
"""

import random
from enum import IntEnum

import torch


class Action(IntEnum):
    """The four controls the decoder chooses between.

    The values are the decoder's output node order, so ``Action(argmax)`` is
    the chosen control.
    """

    LEFT = 0
    RIGHT = 1
    ROTATE_LEFT = 2
    ROTATE_RIGHT = 3


def _rotate(cells, box):
    """Rotate cells 90 degrees clockwise inside their square bounding box.

    Args:
        cells: (row, column) positions inside the box.
        box: Side length of the box.

    Returns:
        The rotated positions, sorted so a shape has one representation.
    """
    return tuple(sorted((col, box - 1 - row) for row, col in cells))


def _build_rotations():
    """Precompute the four rotations of every tetromino.

    Only the spawn layout of each piece is written down; the rest are generated
    by rotating its bounding box.

    Returns:
        A dict of piece name -> tuple of four cell layouts, clockwise.
    """
    spawn = {
        'I': (4, ((1, 0), (1, 1), (1, 2), (1, 3))),
        'O': (2, ((0, 0), (0, 1), (1, 0), (1, 1))),
        'T': (3, ((0, 1), (1, 0), (1, 1), (1, 2))),
        'S': (3, ((0, 1), (0, 2), (1, 0), (1, 1))),
        'Z': (3, ((0, 0), (0, 1), (1, 1), (1, 2))),
        'J': (3, ((0, 0), (1, 0), (1, 1), (1, 2))),
        'L': (3, ((0, 2), (1, 0), (1, 1), (1, 2))),
    }
    rotations = {}
    for name, (box, cells) in spawn.items():
        states = [tuple(sorted(cells))]
        for _ in range(3):
            states.append(_rotate(states[-1], box))
        rotations[name] = tuple(states)
    return rotations


class Piece:
    """A tetromino at a position and rotation on the board.

    Pieces are treated as immutable: :meth:`moved` and :meth:`rotated` return a
    new piece, which lets the game test a candidate placement before accepting
    it.

    Attributes:
        name: Which tetromino this is, e.g. 'T'.
        rotation: Which of the four rotations is active, 0-3.
        row: Board row of the top-left corner of the piece's bounding box.
        col: Board column of that corner.
    """

    BOX_SIZES = {'I': 4, 'O': 2, 'T': 3, 'S': 3, 'Z': 3, 'J': 3, 'L': 3}
    ROTATIONS = _build_rotations()
    NAMES = tuple(ROTATIONS)

    def __init__(self, name, rotation=0, row=0, col=0):
        self.name = name
        self.rotation = rotation
        self.row = row
        self.col = col

    def cells(self):
        """Return the four board cells this piece fills.

        Returns:
            Four (row, column) board positions.
        """
        return [(self.row + r, self.col + c)
                for r, c in self.ROTATIONS[self.name][self.rotation]]

    def moved(self, drow, dcol):
        """Return a copy shifted by the given offset.

        Args:
            drow: Rows to move down.
            dcol: Columns to move right.

        Returns:
            A new Piece; this one is unchanged.
        """
        return Piece(self.name, self.rotation, self.row + drow, self.col + dcol)

    def rotated(self, direction):
        """Return a copy turned one quarter step.

        Args:
            direction: +1 for clockwise, -1 for anticlockwise.

        Returns:
            A new Piece; this one is unchanged.
        """
        return Piece(self.name, (self.rotation + direction) % 4, self.row, self.col)


class TetrisGame:
    """A playfield, a falling piece, and the rules connecting them.

    Attributes:
        board: HEIGHT x WIDTH grid of locked cells; 1 filled, 0 empty.
        piece: The piece currently under the agent's control.
        lines: Rows cleared so far this episode - the score.
        over: Whether the episode has ended by topping out.
    """

    WIDTH = 10
    HEIGHT = 20

    #: Number of binary values in one observation: two planes of the board.
    OBSERVATION_SIZE = 2 * WIDTH * HEIGHT

    def __init__(self, seed=None):
        """Start a new game.

        Args:
            seed: Seed for the piece sequence, for reproducible episodes.
        """
        self._rng = random.Random(seed)
        self.reset()

    def reset(self):
        """Clear the board and spawn the first piece."""
        self.board = [[0] * self.WIDTH for _ in range(self.HEIGHT)]
        self.lines = 0
        self.over = False
        self.piece = self._spawn()

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def _spawn(self):
        """Create a new piece centred at the top of the board.

        Pieces are drawn uniformly at random and the agent never sees what is
        coming - there is no preview and no hold slot.

        Returns:
            The new Piece, which may already be colliding; that is a top-out.
        """
        name = self._rng.choice(Piece.NAMES)
        return Piece(name, row=0, col=(self.WIDTH - Piece.BOX_SIZES[name]) // 2)

    def _fits(self, piece):
        """Check whether a piece can legally occupy its position.

        Args:
            piece: The candidate placement.

        Returns:
            True if every cell is on the board and not already filled.
        """
        for row, col in piece.cells():
            if not (0 <= row < self.HEIGHT and 0 <= col < self.WIDTH):
                return False
            if self.board[row][col]:
                return False
        return True

    def _clear_full_rows(self):
        """Remove every complete row and drop the rows above it down.

        Returns:
            How many rows were removed.
        """
        kept = [row for row in self.board if not all(row)]
        cleared = self.HEIGHT - len(kept)
        self.board = [[0] * self.WIDTH for _ in range(cleared)] + kept
        return cleared

    def step(self, action):
        """Apply one action, then one row of gravity.

        An illegal action - moving into a wall, a rotation that does not fit -
        is ignored. The frame still advances, so the agent pays for it in lost
        time rather than in a special error case.

        Args:
            action: The control chosen by the decoder.

        Returns:
            (lines_cleared, game_over) for this frame. Both are what the reward
            signal is built from.

        Raises:
            RuntimeError: If the game has already ended.
        """
        if self.over:
            raise RuntimeError('step() on a finished game; call reset() first.')

        if action == Action.LEFT:
            candidate = self.piece.moved(0, -1)
        elif action == Action.RIGHT:
            candidate = self.piece.moved(0, +1)
        elif action == Action.ROTATE_LEFT:
            candidate = self.piece.rotated(-1)
        else:
            candidate = self.piece.rotated(+1)

        if self._fits(candidate):
            self.piece = candidate

        return self._apply_gravity()

    def _apply_gravity(self):
        """Drop the piece one row, locking and resolving the board if it lands.

        Returns:
            (lines_cleared, game_over) for this frame.
        """
        dropped = self.piece.moved(1, 0)
        if self._fits(dropped):
            self.piece = dropped
            return 0, False

        for row, col in self.piece.cells():
            self.board[row][col] = 1

        cleared = self._clear_full_rows()
        self.lines += cleared

        self.piece = self._spawn()
        self.over = not self._fits(self.piece)
        return cleared, self.over

    # ------------------------------------------------------------------
    # What the agent sees
    # ------------------------------------------------------------------

    def observation(self):
        """Render the game state as the binary vector the encoder reads.

        This is the design's "20x20 matrix, top half locked blocks, bottom half
        the falling piece", flattened - the encoder flattens it anyway. The two
        halves are kept apart on purpose: the agent has to tell what it can
        still move from what it can no longer change.

        Returns:
            A float tensor of OBSERVATION_SIZE binary values: the locked board
            first, then the falling piece.
        """
        locked = torch.tensor(self.board, dtype=torch.float32).reshape(-1)

        falling = torch.zeros(self.HEIGHT, self.WIDTH)
        if not self.over:
            for row, col in self.piece.cells():
                falling[row, col] = 1.0

        return torch.cat([locked, falling.reshape(-1)])
