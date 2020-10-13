
import numpy as np
from pommerman.envs.v0 import Pomme

from pommerman.agents.our_agents.agent_utils import *
from pommerman.agents import BaseAgent
from pommerman.agents.our_agents.search_agents.evaluation_functions import *
from pommerman.constants import Action


class MinimaxAgent(BaseAgent):

    def __init__(self, evaluation_function=basic_eval, depth=2, num_players=2):
        super(MinimaxAgent, self).__init__()
        self.evaluate = evaluation_function
        self.depth = depth
        self.num_players = num_players
        self.env = None
        self.id = None

    def initialize(self, env : Pomme):
        """
        MUST BE RUN BEFORE USING
        """
        print("initializing")
        self.env = env
        self.id = self._character.agent_id

    def get_actions(self, game_state, agent_num):
        """
        A function that filters actions to possible actions
        :param game_state:
        :param agent_num:
        :return:
        """
        cur_agent_id = (agent_num + self.id) % self.num_players
        pos = game_state.agents[cur_agent_id].position
        ammo = game_state.agents[cur_agent_id].ammo

        is_near_bomb, isOnBomb, _ = bombs_and_me(game_state, cur_agent_id)
        is_stuck = is_pos_stuck(game_state.board, pos)
        res = [a for a in Action if (a.value == 5 and ammo > 0 and not is_stuck) or
               (a.value != 5 and is_valid_direction(game_state.board, pos, a))]
        return res

    def act(self, obs, action_space):
        """
        A function that returns the next action desired by the agents.
        :param obs:
        :param action_space:
        :return:
        """
        game_state = GameState.build_gamestate_from_env(self.env)

        assert self.env is not None
        assert self.id is not None
        # print("tick")
        best_act, value =  self.helper(obs, game_state, self.depth, agent_num=0, curr_actions=[])
        # print("tock")
        # print(f"chosen action: {best_act}, value: {value}")
        return best_act

    def helper(self, obs, game_state, depth, agent_num, curr_actions):
        """
        A function that uses minimax trees to compute the next action .
        :param obs:
        :param game_state:
        :param depth:
        :param agent_num:
        :param curr_actions:
        :return:
        """

        if (depth == 0) or not game_state.agents[self.id].is_alive:
            return Action.Stop.value, self.evaluate(obs, game_state, self.env, agent_num, self.id)

        actions = dict()  # dict of {action: value}

        # decrease depth only if it's the last agent.
        new_depth = depth - 1 if agent_num == (self.num_players - 1) else depth
        next_agent = (agent_num + 1) % self.num_players
        action_space = self.get_actions(game_state, agent_num)
        for action in action_space:
            # print(f"agent_num {agent_num}, action {action}")
            # actions_lst = [action if i == agent_num else Action.Stop for i in range(self.num_players)] # only this agent acts, rest are not doing anything
            if next_agent == 0:
                next_game_state = generate_successor(self.env, curr_actions + [action.value], game_state)
                actions.update(
                    {action.value: self.helper(
                        obs,
                        next_game_state,
                        new_depth,
                        next_agent,
                        [])[1]})
            else:
                actions.update(
                    {action.value: self.helper(
                        obs,
                        game_state,
                        new_depth,
                        next_agent,
                        curr_actions= curr_actions + [action.value])[1]})

        func = max if agent_num == 0 else min
        # print("agens, actions, func", agent_num, actions, func)
        best_value = func(actions.values())

        best_actions = [k for k, v in actions.items() if v == best_value]
        # print("best_act, best_val", best_actions, best_value)
        return np.random.choice(best_actions), best_value


def generate_successor(env, actions, game_state: GameState,agent_id=0):
    """
    A function that given a list of actions returns board, agents, bombs, items, flames after one step
    towards directions
    """
    curr_game_state = copy.deepcopy(game_state)
    actions = np.roll(actions,agent_id)
    next_board, next_agents, next_bombs, next_items, next_flames = env.model.step(actions,
                                                                                  curr_game_state.board,
                                                                                  curr_game_state.agents,
                                                                                  curr_game_state.bombs,
                                                                                  curr_game_state.items,
                                                                                  curr_game_state.flames)
    next_steps_count = curr_game_state.step_count + 1
    return GameState(next_board, next_agents, next_bombs, next_items, next_flames, next_steps_count)



class AlphaBetaAgent(BaseAgent):

    def __init__(self, evaluation_function=balanced_eval, depth=2, num_players=2):
        super(AlphaBetaAgent, self).__init__()
        self.evaluate = evaluation_function
        self.depth = depth
        self.num_players = num_players
        self.env = None
        self.id = None

    def initialize(self, env : Pomme):
        """
        MUST BE RUN BEFORE USING
        """
        self.env = env
        self.id = self._character.agent_id
        self.num_players = len(env._agents)

    # def act(self, obs, action_space):
    #     return self.helper(obs, action_space, self.depth, alpha=-np.inf, beta=np.inf, agent_num=0)[0]
    def act(self, obs, action_space):
        """
        A function that returns the next action desired by the agents.
        :param obs:
        :param action_space:
        :return:
        """
        game_state = GameState(self.env._board,
                               self.env._agents,
                               self.env._bombs,
                               self.env._items,
                               self.env._flames,
                               self.env._step_count)
        assert self.env is not None
        assert self.id is not None
        # print("tick")
        best_act, value = self.helper(obs, game_state, self.depth, agent_num=0, curr_actions=[],
                                      alpha=- np.inf,beta=np.inf)
        # print("tock")
        # print(f"chosen action: {best_act}, value: {value}")

        return best_act.value


    def get_actions(self, game_state, agent_num):
        """
        A function that filters actions to possible actions
        :param game_state:
        :param agent_num:
        :return:
        """
        cur_agent_id = (agent_num + self.id) % self.num_players
        pos = game_state.agents[cur_agent_id].position
        ammo = game_state.agents[cur_agent_id].ammo

        _,isOnBomb,__  = bombs_and_me(game_state,cur_agent_id)

        res = [a for a in Action if (a.value == 5 and ammo > 0 ) or
                        (a.value != 5 and is_valid_direction(game_state.board,pos, a))]
        return res



    def helper(self, obs, game_state, depth, agent_num, curr_actions,alpha,beta):
        """
        A function that uses minimax trees to compute the next action .
        :param obs:
        :param game_state:
        :param depth:
        :param agent_num:
        :param curr_actions:
        :return:
        """

        if (depth == 0) or not game_state.agents[self.id].is_alive:
            return Action.Stop.value, self.evaluate(obs, game_state, self.env,self.id)
        # decrease depth only if it's the last agent.
        new_depth = depth - 1 if agent_num == (self.num_players - 1) else depth
        next_agent = (agent_num + 1) % self.num_players
        action_space = self.get_actions(game_state,agent_num)
        random.shuffle(action_space)
        bestAction = Action.Bomb

        if agent_num == 0:
            for action in action_space:
                _, newAlpha = self.helper(obs,game_state,new_depth,next_agent,
                                         curr_actions=curr_actions + [action.value],
                                         alpha=alpha,beta= beta)
                if alpha < newAlpha:
                    alpha, bestAction = newAlpha, action
                if beta <= alpha:
                    break
            return bestAction, alpha

        else: # not first agent
            if next_agent == 0: # last oponnent agent
                for action in action_space:
                    next_game_state = generate_successor(self.env,
                                                         curr_actions + [action.value],
                                                         game_state,self.id)
                    _, newBeta = self.helper(obs,
                                             next_game_state,
                                             new_depth,
                                             next_agent,
                                             [],
                                             alpha,
                                             beta)
                    if beta > newBeta:
                        beta, bestAction = newBeta, action
                    if beta <= alpha:
                        break
                return bestAction, beta
            else:
                for action in action_space:
                    _, newBeta = self.helper(obs,
                                             game_state,
                                             new_depth,
                                             next_agent,
                                             curr_actions=curr_actions + [action.value],
                                             alpha=alpha,
                                             beta= beta)
                    if beta > newBeta:
                        beta, bestAction = newBeta, action
                    if beta <= alpha:
                        break
                return bestAction, beta





