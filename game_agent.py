"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def centrality(game, move):
    x, y = move
    cx, cy = (math.ceil(game.width / 2), math.ceil(game.height / 2))
    return (x - cx) ** 2 + (y - cy) ** 2


def freedom_from_common_moves(game, player):
    pmoves = game.get_legal_moves()
    omoves = game.get_legal_moves(game.get_opponent(player))
    cmoves = pmoves and omoves
    return 8 - len(cmoves)


def best_common_move_to_center(game, player):
    pmoves = game.get_legal_moves()
    omoves = game.get_legal_moves(game.get_opponent(player))
    cmoves = pmoves and omoves
    if not cmoves:
        return 0
    return max(centrality(game, m) for m in cmoves)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return 0.


def x(game, player):
    opp = game.get_opponent(player)
    opp_moves = game.get_legal_moves(opp)
    p_moves = game.get_legal_moves()
    if not opp_moves:
        return float("inf")
    if not p_moves:
        return float("-inf")
    return float(len(p_moves) - len(opp_moves) + sum(centrality(game, m) for m in p_moves))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return float(freedom_from_common_moves(game, player))


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    ploc_x, ploc_y = game.get_player_location(player)
    oloc_x, oloc_y = game.get_player_location(game.get_opponent(player))
    pmoves = game.get_legal_moves(player)
    omoves = game.get_legal_moves(game.get_opponent(player))

    if math.fabs(ploc_x - oloc_x) >= game.width / 2 and math.fabs(ploc_y - oloc_y) >= game.height / 2:
        return float(len(pmoves) - len(omoves))
    return float(freedom_from_common_moves(game, player))


def custom_score_4(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    ploc_x, ploc_y = game.get_player_location(player)
    oloc_x, oloc_y = game.get_player_location(game.get_opponent(player))

    return float((ploc_x - oloc_x) ** 2 + (ploc_y - oloc_y) ** 2)


def custom_score_5(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    pmoves = len(game.get_legal_moves())
    omoves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(pmoves - omoves + centrality(game, game.get_player_location(player)))


def custom_score_6(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return float(freedom_from_common_moves(game, player) + best_common_move_to_center(game, player))


def custom_score_7(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return 0


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10., name='default-player'):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.name = name

    def active_player(self, game):
        return game.active_player == self;


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return self.minimax_move(game, depth)[0]

    def minimax_move(self, game, depth):
        """
        Original MiniMax function - refactored to return score and move both
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return game.get_player_location(self), self.score(game, self)

        min_or_max_fn, value, best_move = None, None, (-1, -1)

        if self.active_player(game):
            min_or_max_fn, value = max, float("-inf")
        else:
            min_or_max_fn, value = min, float("inf")

        for move in game.get_legal_moves():
            next_ply = game.forecast_move(move)
            score = self.minimax_move(next_ply, depth - 1)[1]
            if min_or_max_fn(value, score) == score:
                best_move = move
                value = score
        return best_move, value


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        for i in range(1, 10000):
            try:
                best_move = self.alphabeta(game, i)
            except SearchTimeout:
                break
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return self.alphabeta_move(game, depth)[0]

    def alphabeta_move(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """
        Original AlphaBeta function - refactored to return score and move both
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return (-1, -1), self.score(game, self)

        if self.active_player(game):
            current_score = float("-inf")
        else:
            current_score = float("inf")

        legal_moves = game.get_legal_moves()
        best_move = legal_moves[0] if len(legal_moves) > 0 else (-1, -1)

        for move in legal_moves:
            next_ply_game = game.forecast_move(move)
            next_ply_score = self.alphabeta_move(next_ply_game, depth - 1, alpha, beta)[1]

            if self.active_player(game):
                if current_score < next_ply_score:
                    current_score = next_ply_score
                    best_move = move
                if current_score >= beta:
                    return best_move, current_score
                else:
                    alpha = max(next_ply_score, alpha)
            else:
                if current_score > next_ply_score:
                    current_score = next_ply_score
                    best_move = move
                if current_score <= alpha:
                    return best_move, current_score
                else:
                    beta = min(next_ply_score, beta)

        if legal_moves and best_move == (-1, -1):
            print ("Phat gaya", legal_moves, " : ", best_move, " : ", depth, " : ", self.active_player(game))

        return best_move, current_score
