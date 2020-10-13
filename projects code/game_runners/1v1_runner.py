'''An example to show how to set up an pommerman game programmatically'''
import os
os.environ['LANG']='en_US'
import sys
sys.path.append('.')
import pickle

from pommerman.agents import ApproximateQAgent



import pommerman
from pommerman import agents
from pommerman.agents.smart_random_agent.smart_random_agent import SmartRandomAgent
from pommerman import characters
NUM_GAMES = 15

def load_agent(filename):
    with open(filename, "rb") as f:
        agent_data = pickle.load(f)
    agent = ApproximateQAgent(extractor="SimpleExtractor")
    agent.weights = agent_data['weights']
    agent.epsilon = agent_data['epsilon']
    agent.alpha = agent_data['alpha']
    agent.discount = agent_data['discount']
    return agent

def main():
    '''
    Simple function to bootstrap a game.
    runs A one v one game and outputs the results into file

    '''
    def pairs_gen(source):
        result = []
        for p1 in range(len(source)):
            for p2 in range(p1 + 1, len(source)):
                result.append([source[p1], source[p2]])
        return result

    def namestr(obj, namespace):
        return [name for name in namespace if namespace[name] is obj]


    alpha_balanced_d2 = agents.AlphaBetaAgent(evaluation_function=agents.balanced_eval,depth=2)
    alpha_balanced_d3 = agents.AlphaBetaAgent(evaluation_function=agents.balanced_eval,depth=3)
    alpha_attacker_d2 = agents.AlphaBetaAgent(evaluation_function=agents.attacker_eval,depth=2)
    alpha_attacker_d3 = agents.AlphaBetaAgent(evaluation_function=agents.attacker_eval,depth=3)
    alpha_coward_d2 = agents.AlphaBetaAgent(evaluation_function=agents.pacifist_eval,depth=2)
    alpha_coward_d3 = agents.AlphaBetaAgent(evaluation_function=agents.pacifist_eval,depth=2)
    smart_1 = SmartRandomAgent()
    simple_1 = agents.SimpleAgent()
    simple_2 = agents.SimpleAgent()
    all_agents_1v1 = [alpha_balanced_d2,alpha_balanced_d3,
                      alpha_attacker_d2,alpha_attacker_d3,
                      alpha_coward_d2,alpha_coward_d3,
                      simple_1]

    file_name = "saved_agents/8000xp_AQL_vs_ab_0_eps_0.5_alpha_0.02_discount_0.8/stats_agent_7900_2020-08-11_04_00_40_355636.pkl"
    trained_agent = load_agent(file_name)

    pair = [alpha_balanced_d2, simple_1]

    p1 , p2 = namestr(pair[0],locals())[0],namestr(pair[1],locals())[0]
    f = open(f"{p1}_vs_{p2}_100_games_str=4",'w')
    agent_list = [
        pair[0],
        pair[1]
    ]
    # Make the "1v1" environment using the agent list
    pair[0]._character = characters.Bomber

    env = pommerman.make('OneVsOne-v0', agent_list)

    print(type(agent_list[0]))
    print(type(agent_list[1]))

    for a in agent_list:
        a.initialize(env)

    wins = [0,0,0]
    for i_episode in range(NUM_GAMES):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

        if env._agents[0].is_alive and env._agents[1].is_alive:
            wins[0] +=1
            print("tie")
        elif env._agents[0].is_alive:
            wins[1] +=1
            print(p1)
        else :
            #env._agents[1].is_alive:
            wins[2] +=1
            print(p2)

        print('Episode {} finished'.format(i_episode))
    f.write(f" ties : {wins[0]}\n player {p1} {wins[1]}\n player {p2} : {wins[2]}")
    print(f" ties : {wins[0]}\n player {p1} {wins[1]}\n player {p2} : {wins[2]}")
    f.close()
    env.close()

    env = None


if __name__ == '__main__':
    main()
