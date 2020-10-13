'''An example to show how to set up an pommerman game programmatically'''
import time

import pommerman
from pommerman import agents


def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''
    with_write = True
    num_rounds = 1

    if with_write:
        num_rounds = 100
        file = open("test_vs_two_simple_depth_str=4_"+str(2)+"__100___" + str(time.time()),'w')

    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)
    a = agents.AlphaBetaAgent(evaluation_function=agents.tomer_eval)
    # Create a set of agents (exactly four)
    agent_list = [
        a,
        # agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),

        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    agent_list[0].initialize(env)
    # Run the episodes just like OpenAI Gym
    for i_episode in range(num_rounds):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

        if with_write:

            if env._agents[0].is_alive and env._agents[1].is_alive and env._agents[1].is_alive:
                file.write("tie\n\n")
            elif env._agents[0].is_alive:
                file.write("agent 0 won\n\n")
            elif env._agents[1].is_alive:
                file.write("agent 1 won\n\n")
            else :
                #env._agents[2].is_alive:
                file.write("agent 2 won\n\n")
        print('Episode {} finished'.format(i_episode))
    if with_write:
        file.close()
    env.close()


if __name__ == '__main__':
    main()
