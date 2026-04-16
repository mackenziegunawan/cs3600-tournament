import math
from game.enums import MoveType, BOARD_SIZE
from game.move import Move


class Searcher:
    """
    Iterative-deepening expectiminimax with alpha-beta pruning.

    The game tree has two types of nodes:
      - Player nodes  (max nodes): we pick the best move
      - Opponent nodes (min nodes): opponent picks worst-for-us move
    
    Search moves are NOT included in the tree — we handle them separately
    in agent.py based on the rat tracker's EV.
    """

    def __init__(self, heuristic_fn, rat_tracker):
        """
        Parameters
        ----------
        heuristic_fn : callable(board, rat_tracker) -> float
        rat_tracker  : RatTracker instance (used in heuristic only)
        """
        self.heuristic = heuristic_fn
        self.rat_tracker = rat_tracker
        self.best_move = None

    def choose_move(self, board, time_budget: float):
        """
        Iterative deepening: deepen until time_budget (seconds) is nearly gone.
        Returns the best Move found.
        """
        import time
        deadline = time.perf_counter() + time_budget

        self.best_move = None
        fallback = self._get_fallback(board)

        for depth in range(1, 10):
            if time.perf_counter() >= deadline - 0.01:
                break
            try:
                move, _ = self._minimax(
                    board, depth, -math.inf, math.inf,
                    maximizing=True, deadline=deadline
                )
                if move is not None:
                    self.best_move = move
            except _TimeUp:
                break

        return self.best_move if self.best_move is not None else fallback

    # ------------------------------------------------------------------
    # Core minimax
    # ------------------------------------------------------------------

    def _minimax(self, board, depth, alpha, beta, maximizing, deadline):
        import time
        if time.perf_counter() >= deadline:
            raise _TimeUp()

        if depth == 0 or board.is_game_over():
            return None, self.heuristic(board, self.rat_tracker)

        moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if not moves:
            return None, self.heuristic(board, self.rat_tracker)

        best_move = moves[0]

        if maximizing:
            best_val = -math.inf
            for move in moves:
                child = board.forecast_move(move, check_ok=False)
                if child is None:
                    continue
                child.reverse_perspective()
                _, val = self._minimax(child, depth - 1, alpha, beta,
                                       maximizing=False, deadline=deadline)
                if val > best_val:
                    best_val = val
                    best_move = move
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break  # prune
            return best_move, best_val
        else:
            best_val = math.inf
            for move in moves:
                child = board.forecast_move(move, check_ok=False)
                if child is None:
                    continue
                child.reverse_perspective()
                _, val = self._minimax(child, depth - 1, alpha, beta,
                                       maximizing=True, deadline=deadline)
                if val < best_val:
                    best_val = val
                    best_move = move
                beta = min(beta, best_val)
                if beta <= alpha:
                    break  # prune
            return best_move, best_val

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_fallback(self, board):
        """Return a safe non-search move, preferring prime > plain."""
        moves = board.get_valid_moves(enemy=False, exclude_search=True)
        # prefer prime moves as a fallback
        for m in moves:
            if m.move_type == MoveType.PRIME:
                return m
        return moves[0] if moves else Move.search((0, 0))


class _TimeUp(Exception):
    pass
