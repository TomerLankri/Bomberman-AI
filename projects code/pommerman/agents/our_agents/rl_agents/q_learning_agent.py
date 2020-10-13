from pommerman import characters
from pommerman.constants import Action

from pommerman.agents.our_agents import agent_utils, course_util
from pommerman.envs.v0 import Pomme

from pommerman.agents import BaseAgent, GameState, generate_successor
from pommerman.agents.our_agents.rl_agents.features_functions import *

import numpy as np


class QLearningAgent(BaseAgent):
    """
    trying to make something like TensorforceAgent
    """

    def __init__(self, alpha=0.5,
                 epsilon=0.5,
                 discount=0.6,
                 episode=0,
                 get_features=get_identity_features,
                 qvals_dict=None,
                 character=characters.Bomber):
        super(QLearningAgent, self).__init__(character)
        self.get_features = get_features
        if qvals_dict is None:
            self.qvals_dict = course_util.Counter()
        else:
            self.qvals_dict = qvals_dict
        self.env = None
        self.id = None
        self.last_action = None
        self.last_state = None
        self.resulting_state = None
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.episode = episode

    def initialize(self, env: Pomme):
        self.env = env
        self.id = self._character.agent_id
        return self

    def getQValue(self, obs, action):  # TODO should we replace obs with gameState?
        """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
        return self.qvals_dict[(obs, action)]

    def get_legal_actions(self, game_state):
        if not game_state.agents[self.id].is_alive:
            return []
        return Action

    def get_value(self, state):
        """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
        actions = self.get_legal_actions(state)
        if len(actions) == 0:
            return 0.0
        return max(self.getQValue(state, action.value) for action in actions)

    def get_policy(self, state: GameState, action_space):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        if len(action_space) == 0:
            return None
        best_actions = self.get_best_actions(action_space, state)

        res = np.random.choice(best_actions)
        return res

    def get_best_actions(self, legal_actions, state: GameState):
        best_actions = []
        best_value = np.NINF
        for action in legal_actions:
            newVal = self.getQValue(state, action.value)
            if newVal > best_value:
                best_actions = [action.value]
                best_value = newVal
            elif newVal == best_value:
                best_actions.append(action.value)
        return best_actions

    def act(self, obs, action_space):
        game_state = GameState.build_gamestate_from_env(self.env)
        self.last_state = game_state
        action_space = self.get_legal_actions(game_state)

        if len(action_space) == 0:
            self.last_action = 0
        elif course_util.flipCoin(self.epsilon):
            self.last_action = np.random.choice(action_space).value
        else:
            self.last_action = self.get_policy(game_state, action_space)
        return self.last_action

    def observe(self, state, reward, done, info):  # TODO use done (says if the game ended)
        reward = reward[self.id]
        q_val = self.getQValue(self.last_state, self.last_action)
        curr_state = GameState.build_gamestate_from_env(self.env)
        self.qvals_dict[(self.last_state, self.last_action)] = q_val + self.alpha * (reward + self.discount * self.get_value(curr_state) - q_val)

    def __hash__(self):
        return hash(self.env.board) + hash(self.env.bombs) + hash(self.env.flames)

    def __eq__(self, other):
        return self.env.board == other.env.board and self.env.bombs == other.env.bombs and self.env.flames == other.env.flames

    def reset(self):
        print("resetting")
        self.last_action = None
        self.last_state = None

    def episode_end(self, reward):
        """This is called at the end of the episode to let the agent know that
        the episode has ended and what is the reward.

        Args:
          reward: The single reward scalar to this agent.
        """
        self.episode += 1


class PommerQAgent(QLearningAgent):

    def __init__(self, epsilon=0.5, alpha=0.02, discount=0.8, numTraining=0.0, **args):
        """
        :param epsilon: exploration prob
        :param alpha: learning rate
        :param discount: discount factor
        :param numTraining:
        :param args: 
        """

        args['epsilon'] = epsilon
        args['alpha'] = alpha
        args['discount'] = discount
        self.numTraining = numTraining
        QLearningAgent.__init__(self, **args)

    def act(self, obs, action_space):
        action = QLearningAgent.act(self, obs, action_space)
        return action


class ApproximateQAgent(PommerQAgent):

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = course_util.lookup(extractor, globals())()
        QLearningAgent.__init__(self, **args)
        self.weights = course_util.Counter()

    def getQValue(self, state, action):
        state = GameState.build_gamestate_from_env(self.env)
        actions = [0] * len(state.agents)
        actions[self.id] = action if action else 0
        next_state = generate_successor(self.env, actions, state)
        return self.weights * self.featExtractor.getFeatures(state, action, self.id, self.env, next_state=next_state)

    def observe(self, obs, reward, done, info):
        next_state = GameState.build_gamestate_from_env(self.env)
        reward = self._custom_reward(next_state, reward, done)
        features = self.featExtractor.getFeatures(self.last_state, self.last_action, self.id, self.env, next_state=next_state)

        for feature, feature_val in features.items():
            difference = reward + self.discount * self.get_value(next_state) - self.getQValue(self.last_state, self.last_action)
            self.weights[feature] = self.weights[feature] + self.alpha * np.clip(feature_val * difference, -1, 1)

    def _custom_reward(self, state, reward, done):
        am_i_alive = int(state.agents[self.id].is_alive)
        some_enemy_alive = any(agent.is_alive for agent_id, agent in enumerate(state.agents) if agent_id != self.id)
        won = am_i_alive and not some_enemy_alive
        lost = not am_i_alive and some_enemy_alive

        if won:
            reward = 100
        elif lost:
            reward = -1000
        else:
            reward = 0
        return reward
