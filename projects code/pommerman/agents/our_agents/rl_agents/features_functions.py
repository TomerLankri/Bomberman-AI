from pommerman.agents import generate_successor
from pommerman.agents.our_agents.search_agents.evaluation_functions import *
from pommerman.agents.our_agents import course_util


def get_identity_features(game_state, action_space, player, env):
    return game_state


class FeatureExtractor:
    def getFeatures(self, state, action, agent_id, env):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        course_util.raiseNotDefined()


class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action, agent_id, env):
        features = course_util.Counter()
        features[(state, action)] = 1.0
        return features


class SimpleExtractor(FeatureExtractor):
    def getFeatures(self, state, action, agent_id, env, next_state=None):

        board_dims = (len(next_state.board) * len(next_state.board[0]))
        closest_agent_dist = find_closest_agent(next_state, agent_id)[0]

        features = course_util.Counter()


        # features["num_bombs"] = len(next_state.bombs)
        # features["number_of_broken_walls"] = get_number_of_broken_walls(next_state, agent_id)
        # features["is_enemy_near_bomb"] = is_closest_enemy_near_bomb(next_state, agent_id)

        features["bias"] = 1.0
        features["bombing_enemy"] = is_bombing_enemy(next_state, agent_id, action)
        features["dist_from_closest_enemy"] = closest_agent_dist / board_dims

        features["is_not_near_bomb"] = int(not (is_agent_near_bomb(next_state, agent_id)))
        features["is_not_in_fire_range"] = int((not (is_agent_in_bomb_range(next_state, agent_id) or is_agent_on_flame(next_state, agent_id))))

        features.divideAll(10.0)

        return features



def is_bombing_enemy(game_state, my_id, action):
    d, agent_id = find_closest_agent(game_state, my_id)
    agent_pos = game_state.agents[agent_id].position
    my_pos = game_state.agents[my_id].position
    if distance.cdist([my_pos], [agent_pos], metric='cityblock')[0][0] == 1 and action == 5:
        return 1
    return 0
