import numpy as np
from game.enums import Cell, BOARD_SIZE

# Noise emission probabilities per cell type
# Indexed by Noise enum: SQUEAK=0, SCRATCH=1, SQUEAL=2
NOISE_EMIT = {
    Cell.BLOCKED: (0.5, 0.3, 0.2),
    Cell.SPACE:   (0.7, 0.15, 0.15),
    Cell.PRIMED:  (0.1, 0.8, 0.1),
    Cell.CARPET:  (0.1, 0.1, 0.8),
}

# Distance error offsets and their probabilities
DIST_ERROR_OFFSETS = (-1, 0, 1, 2)
DIST_ERROR_PROBS   = (0.12, 0.7, 0.12, 0.06)


def _manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


class RatTracker:
    """
    Hidden Markov Model over the 64-cell grid.

    belief[i] = P(rat is at cell i) after all observations so far.
    Cell index: i = y * BOARD_SIZE + x
    """

    def __init__(self, transition_matrix):
        """
        Parameters
        ----------
        transition_matrix : 64x64 list/array
            T[i][j] = P(rat moves from cell i to cell j)
        """
        # Convert to numpy for fast matrix ops
        self.T = np.array(transition_matrix, dtype=np.float64)  # shape (64, 64)

        # Uniform prior — rat ran 1000 steps so we have no useful prior info
        self.belief = np.ones(BOARD_SIZE * BOARD_SIZE, dtype=np.float64)
        self.belief /= self.belief.sum()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, board, noise, estimated_distance, worker_pos):
        """
        Called once per turn BEFORE we choose our move.
        1. Predict: propagate belief through T (rat moved once)
        2. Update:  reweight by noise emission and distance likelihood
        """
        self._predict()
        self._observe_noise(board, noise)
        self._observe_distance(worker_pos, estimated_distance)

    def best_search_cell(self):
        """Return (x, y) of the highest-belief cell."""
        idx = int(np.argmax(self.belief))
        return self._idx_to_pos(idx)

    def search_ev(self, loc):
        """
        Expected value of searching a specific cell.
        EV = belief[cell] * 4 - (1 - belief[cell]) * 2
        """
        idx = self._pos_to_idx(loc)
        p = self.belief[idx]
        return p * 4 - (1 - p) * 2

    def best_search_ev(self):
        """Expected value of searching the single best cell."""
        return self.search_ev(self.best_search_cell())

    def notify_search_result(self, loc, found):
        """
        Called after a search move so we can hard-update the belief.
        If found: rat was at loc (belief becomes a spike, but rat respawns so reset).
        If not found: zero out that cell and renormalize.
        """
        if found:
            # Rat respawns — reset to uniform
            self.belief[:] = 1.0 / (BOARD_SIZE * BOARD_SIZE)
        else:
            idx = self._pos_to_idx(loc)
            self.belief[idx] = 0.0
            total = self.belief.sum()
            if total > 0:
                self.belief /= total

    def notify_opponent_search(self, loc, found):
        """Same as above but for when the opponent searches."""
        self.notify_search_result(loc, found)

    def get_belief_grid(self):
        """Return belief as an 8x8 numpy array (y, x)."""
        return self.belief.reshape(BOARD_SIZE, BOARD_SIZE)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict(self):
        """Multiply belief through the transition matrix."""
        # new_belief[j] = sum_i belief[i] * T[i, j]
        self.belief = self.belief @ self.T
        # Renormalize for numerical stability
        s = self.belief.sum()
        if s > 0:
            self.belief /= s

    def _observe_noise(self, board, noise):
        """Reweight belief by P(noise | cell_type at each cell)."""
        noise_idx = int(noise)  # 0=squeak, 1=scratch, 2=squeal
        weights = np.empty(BOARD_SIZE * BOARD_SIZE, dtype=np.float64)
        for i in range(BOARD_SIZE * BOARD_SIZE):
            x, y = self._idx_to_pos(i)
            cell_type = board.get_cell((x, y))
            weights[i] = NOISE_EMIT[cell_type][noise_idx]
        self.belief *= weights
        s = self.belief.sum()
        if s > 0:
            self.belief /= s

    def _observe_distance(self, worker_pos, estimated_dist):
        """
        Reweight by P(estimated_dist | actual_dist).
        For each cell i, the actual dist is manhattan(worker, cell).
        The observation model: estimated = actual + offset where offset
        is drawn from DIST_ERROR_OFFSETS / DIST_ERROR_PROBS.
        P(estimated | actual) = sum of DIST_ERROR_PROBS where
            actual + offset == estimated (and actual+offset >= 0 always).
        """
        wx, wy = worker_pos
        weights = np.empty(BOARD_SIZE * BOARD_SIZE, dtype=np.float64)
        for i in range(BOARD_SIZE * BOARD_SIZE):
            x, y = self._idx_to_pos(i)
            actual = _manhattan(wx, wy, x, y)
            p = 0.0
            for offset, prob in zip(DIST_ERROR_OFFSETS, DIST_ERROR_PROBS):
                reported = actual + offset
                if reported < 0:
                    reported = 0  # clamped (per spec)
                if reported == estimated_dist:
                    p += prob
            weights[i] = p
        self.belief *= weights
        s = self.belief.sum()
        if s > 0:
            self.belief /= s

    def _pos_to_idx(self, pos):
        return pos[1] * BOARD_SIZE + pos[0]

    def _idx_to_pos(self, idx):
        return (idx % BOARD_SIZE, idx // BOARD_SIZE)
