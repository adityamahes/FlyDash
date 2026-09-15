"""The simplest Tetris that still makes sense as a control problem.

Deliberately left out, to keep the first iteration easy to reason about: no
next-piece preview, no hold slot, no soft drop, no wall kicks, no levels or
speed-up, no multi-line score bonus. Gravity pulls the piece down exactly one
row per frame, so a frame is a fixed unit of time for both the game and the
brain simulation.

``Action.DROP`` is the exception: it slams the piece to its resting place in a
single frame, the space bar of an ordinary Tetris. Without it the agent cannot
choose *when* a piece lands, only nudge it while gravity decides - so every
piece consumed the same ~19 frames and placement quality was nearly invisible
in how long an episode lasted.

Coordinates are (row, column) with row 0 at the top of the board.
"""

import random
from enum import IntEnum

import torch


class Action(IntEnum):
    """The five controls the decoder chooses between.

    The values are the decoder's output node order, so ``Action(argmax)`` is
    the chosen control. ``Decoder`` sizes its output layer from ``len(Action)``,
    so adding a control here widens the decoder automatically.
    """

    LEFT = 0
    RIGHT = 1
    ROTATE_LEFT = 2
    ROTATE_RIGHT = 3
    DROP = 4


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
        applied: Whether the last action actually moved the piece, or was
            rejected as illegal and silently ignored.
        locked: Whether a piece came to rest on the last frame.
        landing_height: Rows above the floor where that piece came to rest.
        eroded_cells: Cells of that piece which a line clear consumed.
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
        # Set by _apply_gravity on the frame a piece comes to rest, so the
        # reward can score the placement that just happened.
        self.locked = False
        self.landing_height = 0
        self.eroded_cells = 0
        self.applied = True
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
        elif action == Action.ROTATE_RIGHT:
            candidate = self.piece.rotated(+1)
        else:
            candidate = self._dropped(self.piece)

        # Recorded so the reward can charge for an action that did nothing:
        # walking into a wall or a rotation that does not fit is ignored here,
        # and without this the agent has no way to be told it wasted a frame.
        self.applied = self._fits(candidate)
        if self.applied:
            self.piece = candidate

        # One row of gravity always follows. After a DROP the piece is already
        # resting, so this is the step that fails to move it and locks it -
        # every placement goes through the same code path.
        return self._apply_gravity()

    def _dropped(self, piece):
        """Find where a piece comes to rest if it falls freely.

        Args:
            piece: The piece to drop.

        Returns:
            The lowest placement that still fits. The piece itself if it is
            already resting on something.
        """
        resting = piece
        while True:
            lower = resting.moved(1, 0)
            if not self._fits(lower):
                return resting
            resting = lower

    def _apply_gravity(self):
        """Drop the piece one row, locking and resolving the board if it lands.

        Returns:
            (lines_cleared, game_over) for this frame.
        """
        self.locked = False
        dropped = self.piece.moved(1, 0)
        if self._fits(dropped):
            self.piece = dropped
            return 0, False

        cells = self.piece.cells()
        for row, col in cells:
            self.board[row][col] = 1

        # Recorded before the rows vanish, because the reward needs to know
        # how much of *this* piece was consumed by the clear.
        full = [row for row in range(self.HEIGHT) if all(self.board[row])]
        self.landing_height = self.HEIGHT - max(row for row, _ in cells)
        self.eroded_cells = len(full) * sum(1 for row, _ in cells if row in full)
        self.locked = True

        cleared = self._clear_full_rows()
        self.lines += cleared

        self.piece = self._spawn()
        self.over = not self._fits(self.piece)
        return cleared, self.over

    # ------------------------------------------------------------------
    # Board quality
    # ------------------------------------------------------------------

    def heights(self):
        """Measure how high the stack reaches in each column.

        Returns:
            One height per column, counted up from the floor. 0 is an empty
            column, HEIGHT a full one.
        """
        heights = []
        for col in range(self.WIDTH):
            height = 0
            for row in range(self.HEIGHT):
                if self.board[row][col]:
                    height = self.HEIGHT - row
                    break
            heights.append(height)
        return heights

    def features(self):
        """Describe the shape of the stack, for the reward to score.

        These are the features classical Tetris players are built on (the
        Dellacherie set). They exist because cleared lines alone say almost
        nothing until the agent is already competent: a board can be one row
        from a clear or riddled with holes and score the same. Each of these
        separates those two cases on every single placement.

        Returns:
            A dict of:
              aggregate_height - total stack height, summed over columns.
              max_height       - the tallest column.
              holes            - empty cells with something filled above them.
                                 The expensive mistake: only a clear above can
                                 free them.
              bumpiness        - total height difference between neighbouring
                                 columns. A jagged surface fits fewer pieces.
              row_transitions  - filled-to-empty changes scanning each row.
                                 Walls count as filled, so a solid row scores 0.
              col_transitions  - the same down each column, floor filled.
              wells            - total depth of one-wide gaps, which only an I
                                 piece can fill.
        """
        heights = self.heights()

        holes = 0
        for col in range(self.WIDTH):
            covered = False
            for row in range(self.HEIGHT):
                if self.board[row][col]:
                    covered = True
                elif covered:
                    holes += 1

        bumpiness = sum(abs(heights[i] - heights[i + 1])
                        for i in range(self.WIDTH - 1))

        row_transitions = 0
        for row in range(self.HEIGHT):
            previous = 1                      # the wall on the left
            for col in range(self.WIDTH):
                cell = self.board[row][col]
                row_transitions += cell != previous
                previous = cell
            row_transitions += previous != 1  # the wall on the right

        col_transitions = 0
        for col in range(self.WIDTH):
            previous = 0                      # open sky above the stack
            for row in range(self.HEIGHT):
                cell = self.board[row][col]
                col_transitions += cell != previous
                previous = cell
            col_transitions += previous != 1  # the floor below it

        wells = 0
        for col in range(self.WIDTH):
            left = heights[col - 1] if col > 0 else self.HEIGHT
            right = heights[col + 1] if col < self.WIDTH - 1 else self.HEIGHT
            depth = min(left, right) - heights[col]
            if depth > 0:
                # A well of depth d needs d cells to fill, and each row of it
                # is harder to reach than the last.
                wells += depth * (depth + 1) // 2

        return {'aggregate_height': sum(heights),
                'max_height': max(heights),
                'holes': holes,
                'bumpiness': bumpiness,
                'row_transitions': row_transitions,
                'col_transitions': col_transitions,
                'wells': wells}

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
