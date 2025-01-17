import numpy as np

from kaggle_helpers import Point


###################
# Helper Function #
###################


def index_to_position(index: int, size: int):
    """
    Converts an index in the observation.halite list to a 2d position in the form (x, y).
    """
    y, x = divmod(index, size)
    return Point(x, (size - y - 1))


# TODO: refactor for vectorized calculation
def cal_dis(x, y):
    """
    Calculate Manhattan Distance for two points
    """
    return sum(abs(x - y))


def estimate_gain(halite, dis, t, collect_rate=0.25, regen_rate=0.02):
    """
    Calculate halite gain for given number of turn.
    """
    if dis >= t:
        return 0
    else:
        # Halite will regenerate before ship arrives
        new_halite = halite * (1 + regen_rate) ** max(0, dis - 1)
        # Ship costs (dis) rounds to arrive destination, uses (t - dis) rounds to collect halite
        return new_halite * (1 - (1 - collect_rate) ** (t - dis))


def unify_pos(pos, size):
    """
    Convert position into standard one.
    Example: Given size = 5, Point(-2, -7) -> Point(3, 3)
    """
    return pos % size


def get_shorter_move(move, size):
    """
    Given one dimension move (x or y), return the shorter move comparing with opposite move.
    The Board is actually round, ship can move to destination by any direction.
    Example: Given board size = 5, move = 3, opposite_move = -2, return -2 since abs(-2) < abs(3).
    """
    if move == 0:
        return 0
    elif move > 0:
        opposite_move = move - size
    else:
        opposite_move = move + size
    return min([move, opposite_move], key=abs)


