#original imports
# from collections.abc import Callable
# from typing import List, Set, Tuple
# import random

# from game import board, move, enums

from collections.abc import Callable
from typing import Tuple
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from game import board as board
from game.move import Move
from game.enums import MoveType, MAX_TURNS_PER_PLAYER, ALLOWED_TIME

from rat_tracker import RatTracker
from heuristic import evaluate
from search import Searcher


# -----------------------------------------------------------------
# Time budget strategy
#
# Total time: 240 seconds over 40 turns = 6 s/turn on average.
# We reserve a small safety buffer and spend the rest on search.
# Early turns get more time (more turns to amortise); later turns
# get a flat floor so we never run out.
# -----------------------------------------------------------------
SAFETY_BUFFER   = 3.0   # never spend below this total remaining
MIN_TURN_BUDGET = 0.3   # floor per turn (seconds)
MAX_TURN_BUDGET = 5.0   # ceiling per turn


class PlayerAgent:
    """
    /you may add and modify functions, however, __init__, commentate and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):

        """
        TODO: Your initialization code below. Should be used to do any setup you want
        before the game begins (i.e. calculating priors.)
        """
        # self.rat_tracker = RatTracker((transition_matrix) if transition_matrix else None)
        # self.searcher = None
        # self._turn = 0

        self.rat_tracker = RatTracker(transition_matrix) if transition_matrix is not None else None
        self.searcher = None
        self._turn = 0

    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        if self.rat_tracker is not None:
            best = self.rat_tracker.best_search_cell()
            ev   = self.rat_tracker.best_search_ev()
            return f"Final rat belief peak: {best}, EV={ev:.3f}"
        return ""

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        """
        TODO: Below is random mover code. Replace it with your own.
        You may do so however you like, including adding extra functions,
        variables. Return a valid move from this function.
        """
        # moves = board.get_valid_moves()w
        # return random.choice(moves)


        noise, estimated_distance = sensor_data

        # 1. Update HMM with this turn's observation
        if self.rat_tracker is not None:
            worker_pos = board.player_worker.get_location()
            self.rat_tracker.update(board, noise, estimated_distance, worker_pos)

            # Incorporate opponent's last search result (if any)
            opp_loc, opp_found = board.opponent_search
            if opp_loc is not None:
                self.rat_tracker.notify_opponent_search(opp_loc, opp_found)

            # Incorporate our own last search result (if any)
            my_loc, my_found = board.player_search
            if my_loc is not None:
                self.rat_tracker.notify_search_result(my_loc, my_found)

        # ---- 2. Decide whether to search ----
        if self.rat_tracker is not None:
            search_ev = self.rat_tracker.best_search_ev()
            best_cell = self.rat_tracker.best_search_cell()
            # Search if EV is clearly positive and we have enough turns left
            turns_remaining = board.player_worker.turns_left
            if search_ev > 1.5 and turns_remaining > 1:
                return Move.search(best_cell)

        # ---- 3. Budget time for this turn ----
        remaining_time  = time_left()
        turns_remaining = board.player_worker.turns_left
        usable_time     = max(0.0, remaining_time - SAFETY_BUFFER)
        # Distribute evenly over remaining turns, clamp to [min, max]
        per_turn = usable_time / max(turns_remaining, 1)
        per_turn = max(MIN_TURN_BUDGET, min(MAX_TURN_BUDGET, per_turn))

        # ---- 4. Run expectiminimax search ----
        if self.searcher is None:
            self.searcher = Searcher(evaluate, self.rat_tracker)
        else:
            # Update the rat_tracker reference in case it changed
            self.searcher.rat_tracker = self.rat_tracker

        best_move = self.searcher.choose_move(board, per_turn)

        return best_move
    
    
