"""
utils relevant for agents, such as helpers to evaluate board state etc
started by adding all helpers from simple_agent
"""
import copy
import pickle
from collections import defaultdict

from pommerman.utility import *


################# our utils #################
class GameState:

    @staticmethod
    def build_gamestate_from_env(env):
        return GameState(env._board,
                         env._agents,
                         env._bombs,
                         env._items,
                         env._flames,
                         env._step_count)

    #TODO: copy in this?
    def __init__(self, board, agents, bombs, items, flames, step_count):
        self.board = board
        self.agents = agents
        self.bombs = bombs
        self.items = items
        self.flames = flames
        self.step_count = step_count

    #TODO: test, compare speed for 2 methods
    def __deepcopy__(self, memodict=None):
        # """method 1:"""
        if memodict is None:
            memodict = {}

        result = GameState.__new__(GameState)
        memodict[id(self)] = result
        result.board = copy.deepcopy(self.board, memo=memodict)
        result.agents = copy.deepcopy(self.agents, memo=memodict)
        result.bombs = copy.deepcopy(self.bombs, memo=memodict)
        result.items = copy.deepcopy(self.items, memo=memodict)
        result.flames = copy.deepcopy(self.flames, memo=memodict)
        result.step_count = self.step_count
        return result

        """method 2 - not working, but can be MUCH faster"""
        # return pickle.loads(pickle.dumps(self, -1))


################# simple_agent's functions #################

def convert_bombs(bomb_map):
    '''Flatten outs the bomb array'''
    ret = []
    locations = np.where(bomb_map > 0)
    for r, c in zip(locations[0], locations[1]):
        ret.append({
            'position': (r, c),
            'blast_strength': int(bomb_map[(r, c)])
        })
    return ret


def filter_invalid_directions(board, my_position, directions, enemies):
    ret = []
    for direction in directions:
        position = get_next_position(my_position, direction)
        if position_on_board(
                board, position) and position_is_passable(
            board, position, enemies):
            ret.append(direction)
    return ret


def directions_in_range_of_bomb(board, my_position, bombs, dist):
    ret = defaultdict(int)

    x, y = my_position
    for bomb in bombs:
        position = bomb['position']
        distance = dist.get(position)
        if distance is None:
            continue

        bomb_range = bomb['blast_strength']
        if distance > bomb_range:
            continue

        if my_position == position:
            # We are on a bomb. All directions are in range of bomb.
            for direction in [
                constants.Action.Right,
                constants.Action.Left,
                constants.Action.Up,
                constants.Action.Down,
            ]:
                ret[direction] = max(ret[direction], bomb['blast_strength'])
        elif x == position[0]:
            if y < position[1]:
                # Bomb is right.
                ret[constants.Action.Right] = max(
                    ret[constants.Action.Right], bomb['blast_strength'])
            else:
                # Bomb is left.
                ret[constants.Action.Left] = max(ret[constants.Action.Left],
                                                 bomb['blast_strength'])
        elif y == position[1]:
            if x < position[0]:
                # Bomb is down.
                ret[constants.Action.Down] = max(ret[constants.Action.Down],
                                                 bomb['blast_strength'])
            else:
                # Bomb is down.
                ret[constants.Action.Up] = max(ret[constants.Action.Up],
                                               bomb['blast_strength'])
    return ret


def is_adjacent_enemy(items, dist, enemies):
    for enemy in enemies:
        for position in items.get(enemy, []):
            if dist[position] == 1:
                return True
    return False


def has_bomb(obs):
    return obs['ammo'] >= 1


def is_safe_to_bomb(ammo, blast_strength, items, dist, my_position):
    """Returns whether we can safely bomb right now.

    Decides this based on:
    1. Do we have ammo?
    2. If we laid a bomb right now, will we be stuck?
    """
    # Do we have ammo?
    if ammo < 1:
        return False

    # Will we be stuck?
    x, y = my_position
    for position in items.get(constants.Item.Passage):
        if dist[position] == np.inf:
            continue

        # We can reach a passage that's outside of the bomb strength.
        if dist[position] > blast_strength:
            return True

        # We can reach a passage that's outside of the bomb scope.
        position_x, position_y = position
        if position_x != x and position_y != y:
            return True

    return False


def nearest_position(dist, objs, items, radius):
    nearest = None
    dist_to = max(dist.values())

    for obj in objs:
        for position in items.get(obj, []):
            d = dist[position]
            if d <= radius and d <= dist_to:
                nearest = position
                dist_to = d

    return nearest
