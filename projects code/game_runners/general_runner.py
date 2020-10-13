'''An example to show how to set up an pommerman game programmatically'''
import os
os.environ['LANG']='en_US'
import sys
sys.path.append('.')

from pommerman.constants import Result

sys.path.append('.')
import pommerman
from pommerman import agents

from pommerman.agents import *
from pommerman.agents.smart_random_agent.smart_random_agent import SmartRandomAgent

NUM_GAMES = 10
DEPTH = 3


def main():
    '''Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''
    print(f"num games: {NUM_GAMES}, 1v1 free-for-all game, DEPTH={DEPTH}")

    alpha_balanced = agents.AlphaBetaAgent(evaluation_function=agents.balanced_eval, depth=DEPTH)
    alpha_attacker = agents.AlphaBetaAgent(evaluation_function=agents.attacker_eval, depth=DEPTH)
    alpha_coward = agents.AlphaBetaAgent(evaluation_function=agents.pacifist_eval, depth=DEPTH)
    agent_list = [
        alpha_balanced,
        alpha_attacker,
        # alpha_coward
        # minimax_agent,
        # SmartRandomAgent(),
        # agents.SimpleAgent(),
        # agents.SimpleAgent(),
        # PlayerAgent(),
    ]
    # Make the "1v1" environment using the agent list
    env = pommerman.make('OneVsOne-v0', agent_list)
    # env = pommerman.make('FFA-3_players-v0', agent_list)
    # env = pommerman.make('PommeFFACompetition-v0', agent_list)
    for agent in agent_list:
        agent.initialize(env)
    # Run the episodes just like OpenAI Gym
    wins = [0] + [0] * len(agent_list)
    for i_episode in range(NUM_GAMES):
        state = env.reset()
        done = False
        info = None
        turns = 0
        while not done:
            turns += 1
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

        if info['result'] == Result.Tie:
            wins[0] += 1
        else:
            wins[info["winners"][0]] += 1
        print(f'Episode {i_episode} finished, info: {info}, took {turns} turns')

    print(f"ties : {wins[0]}\n player zero {wins[1]}\n player one : {wins[2]}, overall games: {NUM_GAMES}")

    env.close()


if __name__ == '__main__':
    main()
