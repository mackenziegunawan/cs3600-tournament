import numpy as np
from game.enums import (
    Cell, Direction, MoveType, BOARD_SIZE, CARPET_POINTS_TABLE
)
from game.move import Move

# Tunable weights
W_SCORE_DIFF    = 2.0   # raw score advantage
W_CARPET_POT    = 0.6   # potential carpet points reachable from current pos
W_PRIME_COUNT   = 0.2   # number of primed squares we "own" (adjacent run)
W_RAT_EV        = 1.2   # expected value of best search available
W_MOBILITY      = 0.05  # number of valid non-search moves (avoid being cornered)
W_TURNS_LEFT    = 0.0   # turns left scaling (usually not needed)

DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


def evaluate(board, rat_tracker) -> float:
    """
    Static evaluation from the perspective of board.player_worker.
    Higher = better for us.
    """
    my_score   = board.player_worker.get_points()
    opp_score  = board.opponent_worker.get_points()
    score_diff = my_score - opp_score

    carpet_pot = _carpet_potential(board)
    prime_cnt  = _best_primeable_run(board)
    rat_ev     = rat_tracker.best_search_ev()
    mobility   = len(board.get_valid_moves(enemy=False, exclude_search=True))

    return (
        W_SCORE_DIFF  * score_diff  +
        W_CARPET_POT  * carpet_pot  +
        W_PRIME_COUNT * prime_cnt   +
        W_RAT_EV      * rat_ev      +
        W_MOBILITY    * mobility
    )


# ------------------------------------------------------------------
# Component helpers
# ------------------------------------------------------------------

def _carpet_potential(board) -> float:
    """
    For each direction from current position, count the longest contiguous
    run of primed cells we could carpet right now, and sum the point values.
    """
    pos = board.player_worker.get_location()
    total = 0.0
    for direction in DIRECTIONS:
        run = 0
        cur = pos
        for _ in range(BOARD_SIZE - 1):
            nxt = _step(cur, direction)
            if nxt is None:
                break
            if board.get_cell(nxt) != Cell.PRIMED:
                break
            # Make sure neither worker occupies it
            if nxt == board.player_worker.get_location():
                break
            if nxt == board.opponent_worker.get_location():
                break
            run += 1
            cur = nxt
        if run > 0:
            total += CARPET_POINTS_TABLE.get(run, 0)
    return total


# def _adjacent_prime_run(board) -> int:
#     """
#     Count how many primed squares are reachable from the current position
#     within 2 steps (rough measure of local primed density we can exploit).
#     """
#     pos = board.player_worker.get_location()
#     count = 0
#     seen = {pos}
#     frontier = [pos]
#     for _ in range(2):
#         next_frontier = []
#         for cur in frontier:
#             for direction in DIRECTIONS:
#                 nxt = _step(cur, direction)
#                 if nxt is None or nxt in seen:
#                     continue
#                 seen.add(nxt)
#                 cell = board.get_cell(nxt)
#                 if cell == Cell.PRIMED:
#                     count += 1
#                     next_frontier.append(nxt)
#                 elif cell == Cell.SPACE:
#                     next_frontier.append(nxt)
#         frontier = next_frontier
#     return count


def _best_primeable_run(board) -> float:
    """
    In each direction from current square, count how many consecutive SPACE
    squares exist (ignoring the worker's own square). This estimates how long
    a prime run we could build if we start priming now. Returns the best
    direction's run length, capped at 6.
    """
    pos = board.player_worker.get_location()
    opp = board.opponent_worker.get_location()
    # Can only prime if current square is SPACE
    if board.get_cell(pos) != Cell.SPACE:
        return 0.0
    best = 0
    for direction in DIRECTIONS:
        run = 0
        cur = pos
        for _ in range(BOARD_SIZE - 1):
            nxt = _step(cur, direction)
            if nxt is None or nxt == opp:
                break
            cell = board.get_cell(nxt)
            if cell != Cell.SPACE:
                break
            run += 1
            cur = nxt
        best = max(best, run)
    return min(best, 6)
 
# a small reward just for standing on a plain square, which discourages the bot from wandering onto carpet/primed squares where it can't prime.
def _can_prime_bonus(board) -> float:
    """
    Return 1.0 if the worker is currently standing on a plain SPACE square
    (i.e. a prime move is available), 0 otherwise. This nudges the bot to
    prefer positioning on plain squares over carpet/primed squares.
    """
    pos = board.player_worker.get_location()
    return 1.0 if board.get_cell(pos) == Cell.SPACE else 0.0

def _step(pos, direction):
    x, y = pos
    if direction == Direction.UP:
        y -= 1
    elif direction == Direction.DOWN:
        y += 1
    elif direction == Direction.LEFT:
        x -= 1
    elif direction == Direction.RIGHT:
        x += 1
    if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
        return (x, y)
    return None
