'''An example to show how to set up an pommerman game programmatically'''
import os

from game_runners.q_learner_trainer import load_agent

os.environ['LANG']='en_US'
import sys
sys.path.append('.')
import pommerman
from pommerman import agents
from pommerman.agents import *
from pommerman.agents.smart_random_agent.smart_random_agent import SmartRandomAgent

NUM_GAMES = 1
DEPTH = 3


def main():

    print(f"num games: {NUM_GAMES}, 1v1 smart vs balanced, DEPTH={DEPTH}")

    alpha_balanced = agents.AlphaBetaAgent(evaluation_function=agents.balanced_eval, depth=DEPTH)
    alpha_attacker = agents.AlphaBetaAgent(evaluation_function=agents.attacker_eval, depth=DEPTH)
    alpha_coward = agents.AlphaBetaAgent(evaluation_function=agents.pacifist_eval, depth=DEPTH)
    file_name = "saved_agents/8000xp_AQL_vs_ab_0_eps_0.5_alpha_0.02_discount_0.8/stats_agent_7900_2020-08-11_04_00_40_355636.pkl"
    trained_agent = load_agent(file_name)
    agent_list = [
        alpha_balanced,
        # alpha_attacker,
        # alpha_coward
        # trained_agent,
        PlayerAgent(),
    ]
    # Make the "1v1" environment using the agent list
    env = pommerman.make('OneVsOne-v0', agent_list)
    # env = pommerman.make('FFA-3_players-v0', agent_list)
    # env = pommerman.make('PommeFFACompetition-v0', agent_list)
    for agent in agent_list:
        agent.initialize(env)
    for i_episode in range(NUM_GAMES):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

        print(f'Episode {i_episode} finished, info: {info}')
        if env._agents[0].is_alive and env._agents[1].is_alive:
            print(f"Game ended with a tie!")
        elif env._agents[0].is_alive:
            print("AlphaBeta agent won!")
        else :
            print("Human player won!")
    env.close()


if __name__ == '__main__':
    main()
