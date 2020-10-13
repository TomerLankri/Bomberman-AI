import math

from scipy.spatial import distance
from pommerman.constants import Action

from pommerman.agents.our_agents.agent_utils import *


def basic_eval(obs, game_state: GameState, env, my_id):
    """
    score = A * num_dead_enemies + B * my_score - C * sum(enemies_score) - D * am_i_dead
    this evaluation treats other agents like their only goal is to kill us!
    """
    A = 2
    B = 4
    C = 1
    D = 10
    E = 1
    num_dead_enem = sum(
        1 for num, agent in enumerate(game_state.agents) if (agent.is_alive and num != my_id))
    rewards = env.model.get_rewards(game_state.agents,
                                    env._game_type,
                                    game_state.step_count,
                                    env._max_steps)  # list of each agent's score
    my_score = rewards.pop(my_id)
    enem_score = sum(rewards)
    am_i_dead = int(not game_state.agents[my_id].is_alive)
    num_bombs = len(game_state.bombs)
    return A * num_dead_enem + B * my_score - C * enem_score - D * am_i_dead + E * num_bombs


def move_eval(obs, game_state: GameState, env, my_id):
    """
    score = A * num_dead_enemies + B * my_score - C * sum(enemies_score) - D * am_i_dead
    this evaluation treats other agents like their only goal is to kill us!
    """
    A = 2
    B = 4
    C = 1
    D = 100
    E = 0
    F = 1
    G = 1

    num_dead_enem = sum(
        1 for num, agent in enumerate(game_state.agents) if (agent.is_alive and num != my_id))
    rewards = env.model.get_rewards(game_state.agents,
                                    env._game_type,
                                    game_state.step_count,
                                    env._max_steps)  # list of each agent's score
    my_score = rewards.pop(my_id)
    enem_score = sum(rewards)
    am_i_dead = int(not game_state.agents[my_id].is_alive)
    num_bombs = len(game_state.bombs)

    is_there_a_bomb_next_to_wall = 0

    isOnFlame = is_agent_on_flame(game_state, my_id)
    is_near_bomb, is_on_bomb, smallest_dist_from_bomb = bombs_and_me(game_state, my_id)
    bombNearWall = get_number_of_broken_walls(game_state, my_id)

    distanceFromEnemy = dist(game_state.agents[0].position, game_state.agents[1].position)

    return A * num_dead_enem + B * my_score + E * num_bombs + G * is_there_a_bomb_next_to_wall - \
           C * enem_score - D * am_i_dead - F * is_near_bomb


def dist(first, second):
    """
    A function
    :param first:
    :param second:
    :return:
    """
    x1, y1 = first
    x2, y2 = second
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_closest_agent(game_state, my_id):
    """
    A function that finds the closest agent to an agent with id my_id
    :param game_state:
    :param my_id:
    :return:
    """
    # gives you back the id of the closest agent
    my_pos = game_state.agents[my_id].position
    agents_dists_ids = [(distance.euclidean(a.position, my_pos), a_id) for a_id, a in
                        enumerate(game_state.agents) if a_id != my_id]
    d, agent_id = min(agents_dists_ids, key=lambda x: x[0])
    return d, agent_id


def is_agent_on_flame(game_state, my_id):
    """
    A function that returns True if an agent with my_id is on a flame
    :param game_state:
    :param my_id:
    :return:
    """
    for flame in game_state.flames:
        if game_state.agents[my_id].position == flame.position:
            return 1
    return 0


BOMB_POS_IDX = 0
BOMB_DIST_IDX = 1


def bombs_and_me(game_state, my_id):
    """
    A function that returns 3 parameters. the first is if an agents is in bomb range, secons is if an agent is on a bomb
    and the third returns the smalles distance from a bomb.
    :param game_state:
    :param my_id:
    :return:
    """
    # if near / on bomb
    is_near_bomb = False
    is_on_bomb = False
    if len(game_state.bombs) == 0:
        return is_near_bomb, is_on_bomb, 0
    my_pos = game_state.agents[my_id].position
    my_x, my_y = my_pos
    bomb_dist_arr = [my_pos, np.inf]

    for bomb in game_state.bombs:
        bomb_x = bomb.position[0]
        bomb_y = bomb.position[1]

        if dist(my_pos, bomb.position) < bomb_dist_arr[BOMB_DIST_IDX]:
            bomb_dist_arr[BOMB_POS_IDX] = bomb.position
            bomb_dist_arr[BOMB_DIST_IDX] = dist(my_pos, bomb.position)
        if my_x == bomb_x and my_y == bomb_y:
            is_on_bomb = True
        if is_agent_in_single_bomb_range(my_x, my_y, bomb):
            is_near_bomb = True
    return is_near_bomb, is_on_bomb, bomb_dist_arr[BOMB_DIST_IDX]


def get_number_of_broken_walls(game_state, my_id):
    """
    A function that retunrs the number of walls broken by an agent in the board.
    :param game_state:
    :param my_id:
    :return:
    """
    result = 0
    my_x, my_y = game_state.agents[my_id].position

    # get all walls off board
    walls = np.where(game_state.board == 2)
    # make sure bounds are OK
    start_x, start_y = max(my_x - 2, 0), max(my_y - 2, 0)
    end_x, end_y = min(my_x + 2, constants.BOARD_SIZE), min(my_x + 2, constants.BOARD_SIZE)

    # check if there is a bomb in range and near a wall
    for bomb in game_state.bombs:
        bomb_x = bomb.position[0]
        bomb_y = bomb.position[1]

        # check if we have put a bomb - we want to reward him ONLY FOR  BOMBS NEAR HIM
        if start_x <= bomb_x <= end_x and start_y <= bomb_y <= end_y:
            # we have a bomb in range, lets check to see if there is a wall near that bomb:
            endBombX, endBombY, startBombX, startBombY = calc_flames(bomb)

            for wallX, wallY in zip(walls[0], walls[1]):
                if (wallX == bomb_x and startBombY <= wallY <= endBombY) or \
                        (wallY == bomb_y and startBombX <= wallX <= endBombX):
                    result += 1
    return result


def is_agent_in_bomb_range(state, agent):
    """
    A function that returns true if an agent is in some  bomb range.
    :param state:
    :param agent:
    :return:
    """
    agent_x = state.agents[agent].position[0]
    agent_y = state.agents[agent].position[1]
    for bomb in state.bombs:
        if is_agent_in_single_bomb_range(agent_x, agent_y, bomb):
            return True
    return False


def is_agent_in_single_bomb_range(agent_x, agent_y, bomb):
    """
    A function that returns true if an agent is in specific bomb range.
    :param state:
    :param agent:
    :return:
    """
    bomb_x = bomb.position[0]
    bomb_y = bomb.position[1]
    endBombX, endBombY, startBombX, startBombY = calc_flames(bomb)
    if bomb_y == agent_y:
        if startBombX <= bomb_x <= endBombX:
            return True
    if bomb_x == agent_x:
        if startBombY <= bomb_y <= endBombY:
            return True


def calc_flames(b):
    """
    A function that calculates the range of the bomb.
    :param b:
    :return:
    """
    # calc flames of this bomb
    strength = b.blast_strength - 1
    startBombX, startBombY = max(b.position[0] - strength, 0), max(b.position[1] - strength, 0)
    endBombX = min(b.position[0] + strength + 1, constants.BOARD_SIZE)
    endBombY = min(b.position[1] + strength + 1, constants.BOARD_SIZE)
    return endBombX, endBombY, startBombX, startBombY


def is_closest_enemy_near_bomb(game_state, my_id):
    """
    A function that returns true if an enemy is in bomb range
    :param game_state:
    :param my_id:
    :return:
    """
    d, agent_id = find_closest_agent(game_state, my_id)
    agent_pos = game_state.agents[agent_id].position
    for bomb in game_state.bombs:
        bomb_pos = bomb.position
        if distance.cdist([bomb_pos], [agent_pos], metric='cityblock')[0][0] <= 2:
            return 1
    return 0


def get_dist_from_closest_bomb(game_state, agent_id):
    """
    A function that returns the distance from the closest bomb.
    :param game_state:
    :param agent_id:
    :return:
    """
    agent_pos = game_state.agents[agent_id].position
    dists = [distance.euclidean(agent_pos, bomb.position) for bomb in game_state.bombs]
    result = 0
    if dists:
        result = min(dists)
    return result


def get_dist_from_closest_flame(game_state, agent_id):
    """
    A function that returns the distance from the closest bomb.
    :param game_state:
    :param agent_id:
    :return:
    """
    agent_pos = game_state.agents[agent_id].position
    dists = [distance.euclidean(agent_pos, flame.position) for flame in game_state.flames]
    result = 0
    if dists:
        result = min(dists)
    return result


def is_agent_on_bomb(game_state, agent_id):
    """
    A function that returns true if an agent is on a bomb.
    :param game_state:
    :param agent_id:
    :return:
    """
    agent_pos = game_state.agents[agent_id].position
    agent_x, agent_y = agent_pos
    for bomb in game_state.bombs:
        bomb_x = bomb.position[0]
        bomb_y = bomb.position[1]
        if agent_x == bomb_x and agent_y == bomb_y:
            return 1
    return 0


def get_actions_for_agent(agent_id, game_state):
    """
    A function that filters actions possible for a given agent.
    :param agent_id:
    :param game_state:
    :return:
    """
    pos = game_state.agents[agent_id].position
    ammo = game_state.agents[agent_id].ammo

    is_near_bomb, isOnBomb, _ = bombs_and_me(game_state, agent_id)
    is_stuck = is_pos_stuck(game_state.board, pos)
    res = [a for a in Action if (a.value == 5 and ammo > 0 and not (is_stuck and is_near_bomb)) or
           (a.value != 5 and is_valid_direction(game_state.board, pos, a))]
    return res


def is_pos_stuck(board, pos):
    """
    A function that detects if an agent is in a stuck position meaning cant move in no direction.
    :param board:
    :param pos:
    :return:
    """
    x, y = pos
    LEN = len(board)
    bool = board[(x + 1) % LEN][y] > 0 and board[(x - 1) % LEN][y] > 0 and board[x][(y + 1) % LEN] > 0 and \
           board[x][(y - 1) % LEN] > 0
    return 1 if bool else 0


def is_agent_near_bomb(game_state, agent_id):
    """
    A function that returns true if an agent is in bomb range.
    :param game_state:
    :param agent_id:
    :return:
    """
    is_near_bomb = False
    my_pos = game_state.agents[agent_id].position
    agent_x, agent_y = my_pos
    for bomb in game_state.bombs:
        bomb_x = bomb.position[0]
        bomb_y = bomb.position[1]
        if (agent_x == bomb_x and (agent_y == bomb_y - 1 or agent_y == bomb_y + 1)) or \
                (agent_y == bomb_y and (agent_x == bomb_x - 1 or agent_x == bomb_x + 1)):
            is_near_bomb = True
    return int(is_near_bomb)


def attacker_eval(obs, game_state: GameState, env, agent_id):
    """
    result = A * num_bombs \
             + C * is_there_a_bomb_next_to_wall \
             - B * is_near_bomb \
             - D * is_on_bomb \
             - E * (is_on_flame + am_i_dead + is_stuck) \
             - G * int(smallest_dist_from_bomb) \
             - F * int(distance_from_enemy) \
             + H * bomb_near_enemy \
             + I * number_of_broken_walls\
             - J * num_enem_actions
     """
    A = 1000
    B = 3
    C = 10
    D = 1
    E = 1000
    F = 2
    G = 1
    H = 3
    I = 1
    J = 0
    return get_value(A, B, C, D, E, F, G, H, I, J, agent_id, game_state)


def balanced_eval(obs, game_state: GameState, env, agent_id):
    """
    result = A * num_bombs \
             + C * is_there_a_bomb_next_to_wall \
             - B * is_near_bomb \
             - D * is_on_bomb \
             - E * (is_on_flame + am_i_dead + is_stuck) \
             - G * int(smallest_dist_from_bomb) \
             - F * int(distance_from_enemy) \
             + H * bomb_near_enemy \
             + I * number_of_broken_walls\
             - J * num_enem_actions
     """
    A = 1
    B = 5
    C = 1
    D = 1
    E = 5
    F = 1
    G = 1
    H = 1
    I = 0
    J = 0
    return get_value(A, B, C, D, E, F, G, H, I, J, agent_id, game_state)


def pacifist_eval(obs, game_state: GameState, env, agent_id):
    """
    result = A * num_bombs \
             + C * is_there_a_bomb_next_to_wall \
             - B * is_near_bomb \
             - D * is_on_bomb \
             - E * (is_on_flame + am_i_dead + is_stuck) \
             - G * int(smallest_dist_from_bomb) \
             - F * int(distance_from_enemy) \
             + H * bomb_near_enemy \
             + I * number_of_broken_walls\
             - J * num_enem_actions
     """
    A = -100
    B = 5
    C = 1
    D = 1
    E = -100
    F = -10
    G = 1
    H = 1
    I = 0
    J = 0.5
    return get_value(A, B, C, D, E, F, G, H, I, J, agent_id, game_state)


def get_value(A, B, C, D, E, F, G, H, I, J, agent_id, game_state):
    """
    A function that returns a value given paramterers
    :param A:
    :param B:
    :param C:
    :param D:
    :param E:
    :param F:
    :param G:
    :param H:
    :param I:
    :param J:
    :param agent_id:
    :param game_state:
    :return:
    """
    am_i_dead, distance_from_enemy, is_near_bomb, is_on_bomb, is_on_flame, is_stuck, is_there_a_bomb_next_to_wall, num_bombs, number_of_broken_walls, smallest_dist_from_bomb, num_enem_actions = get_parameters(
        agent_id, game_state)
    bomb_near_enemy = is_closest_enemy_near_bomb(game_state, agent_id)
    result = A * num_bombs \
             + C * is_there_a_bomb_next_to_wall \
             - B * is_near_bomb \
             - D * is_on_bomb \
             - E * (is_on_flame + am_i_dead + is_stuck) \
             - G * int(smallest_dist_from_bomb) \
             - F * int(distance_from_enemy) \
             + H * bomb_near_enemy \
             + I * number_of_broken_walls \
             - J * num_enem_actions
    return result


def get_parameters(agent_id, game_state):
    """
    A function that computes all the features relevant to the evaluation function regarding a gamestate and an agent
    :param agent_id:
    :param game_state:
    :return:
    """
    am_i_dead = int(not game_state.agents[agent_id].is_alive)
    num_bombs = len(game_state.bombs)
    number_of_broken_walls = get_number_of_broken_walls(game_state, agent_id)
    is_there_a_bomb_next_to_wall = int(number_of_broken_walls > 0)
    is_on_flame = is_agent_on_flame(game_state, agent_id)
    smallest_dist_from_bomb = get_dist_from_closest_bomb(game_state, agent_id)
    is_on_bomb = is_agent_on_bomb(game_state, agent_id)
    is_near_bomb = is_agent_near_bomb(game_state, agent_id)
    distance_from_enemy = find_closest_agent(game_state, agent_id)[0]
    is_stuck = is_pos_stuck(game_state.board, game_state.agents[agent_id].position)
    num_enem_actions = 0 # sum(len(get_actions_for_agent(i, game_state)) for i in range(len(game_state.agents)) if id != agent_id)
    return am_i_dead, \
           distance_from_enemy, \
           is_near_bomb, \
           is_on_bomb, \
           is_on_flame, \
           is_stuck, \
           is_there_a_bomb_next_to_wall, \
           num_bombs, \
           number_of_broken_walls, \
           smallest_dist_from_bomb, \
           num_enem_actions
