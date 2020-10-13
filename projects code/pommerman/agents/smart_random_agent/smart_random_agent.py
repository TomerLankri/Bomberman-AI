'''
# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

An agent that preforms a random action each step
'''
from pommerman.agents.smart_random_agent import action_prune
from pommerman.constants import Action
from pommerman.agents import  BaseAgent
import random

class RandomAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def act(self, obs, action_space):
        return action_space.sample()


class StaticAgent(BaseAgent):
    """ Static agent"""

    def act(self, obs, action_space):
        return Action.Stop.value

class SmartRandomAgent(BaseAgent):
    """ random with filtered actions"""
    def act(self, obs, action_space):
        valid_actions= action_prune.get_filtered_actions(obs)
        if len(valid_actions) ==0:
            valid_actions.append(Action.Stop.value)
        return random.choice(valid_actions)

    def set_start_position(self, start_position):
        self._character.start_position = start_position

class SmartRandomAgentNoBomb(BaseAgent):
    """ random with filtered actions but no bomb"""
    def act(self, obs, action_space):
        valid_actions= action_prune.get_filtered_actions(obs)
        if Action.Bomb.value in valid_actions:
            valid_actions.remove(Action.Bomb.value)
        if len(valid_actions) == 0:
            valid_actions.append(Action.Stop.value)
        return random.choice(valid_actions)
