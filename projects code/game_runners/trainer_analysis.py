import sys
sys.path.append('.')
import os
os.environ['LANG']='en_US'
import matplotlib.pyplot as plt
import pommerman
from game_runners.q_learner_trainer import train, load_agent, run_games
from pommerman.agents import ApproximateQAgent, SimpleAgent, AlphaBetaAgent, attacker_eval, pacifist_eval
from pommerman.agents.smart_random_agent.smart_random_agent import SmartRandomAgent


def populate_success_pct_per_training(num_trainings, checkpoint_step, num_games_to_sample, dir_name="saved_agents"):
    import pandas as pd
    df = pd.DataFrame({'num_training': [], 'success_rate': []})
    # pwd = os.path.dirname(os.path.realpath(__file__))
    dir = dir_name #os.path.join(pwd, dir_name)
    for i in range(0, num_trainings, checkpoint_step):
        file_name = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and 'stats_agent_{}'.format(i) in f][0]
        full_path = os.path.join(dir, file_name)
        training_agent = load_agent(full_path)

        agent_list = [
            # SmartRandomAgent(),
            # SimpleAgent(),
            a_b_agent_balanced,
            # a_b_agent_attck,
            # a_b_agent_pacifist,
            training_agent,
        ]
        # env = pommerman.make('FFA-3_players-v0', agent_list)
        # env = pommerman.make('PommeFFACompetition-v0', agent_list)
        env = pommerman.make('OneVsOne-v0', agent_list)
        for agent in agent_list:
            agent.initialize(env)

        print(f"starting {num_games_to_sample} games with {file_name} agent")
        success_rate = run_games(env, training_agent, num_games=num_games_to_sample, show_games=False)
        df = df.append(pd.DataFrame({'num_training': [i], 'success_rate': [success_rate]}), ignore_index=True)

    return df

depth = 2
a_b_agent_balanced = AlphaBetaAgent(depth=depth)
a_b_agent_attck = AlphaBetaAgent(depth=depth, evaluation_function=attacker_eval)
a_b_agent_pacifist = AlphaBetaAgent(depth=depth, evaluation_function=pacifist_eval)

if __name__ == '__main__':
    params = [
        [0.5, 0.02, 0.8],
    ]
    for i, p in enumerate(params):
        epsilon, alpha, discount = p
        file_name = f'fogel_1_eps_{epsilon}_alpha_{alpha}_discount_{discount}'

        # ==============================================================================================================
        # Init Game
        # ==============================================================================================================

        training_agent = ApproximateQAgent(extractor="SimpleExtractor", epsilon=epsilon, alpha=alpha, discount=discount)

        agent_list = [
            # SmartRandomAgent(),
            # SimpleAgent(),
            a_b_agent_balanced,
            # a_b_agent_attck,
            # a_b_agent_pacifist,
            training_agent,
        ]
        # env = pommerman.make('FFA-3_players-v0', agent_list)
        # env = pommerman.make('PommeFFACompetition-v0', agent_list)
        env = pommerman.make('OneVsOne-v0', agent_list)

        for agent in agent_list:
            agent.initialize(env)
        # training_agent.initialize(env)
        # a_b_agent1.initialize(env)

        # ==============================================================================================================
        # Train Agent
        # ==============================================================================================================
        num_games = 8000
        checkpoint_step = 50
        dir_name = f"fogel_agent_stats_{i}_{file_name}"
        train(env, training_agent, num_games=num_games, save=True, checkpoint_step=checkpoint_step, dir_name=dir_name)

        # ==============================================================================================================
        # Run Stats
        # ==============================================================================================================
        df = populate_success_pct_per_training(num_trainings=num_games,
                                               checkpoint_step=checkpoint_step,
                                               num_games_to_sample=30,
                                               dir_name=dir_name)
        df.to_csv(f"{file_name}.csv")

        # ==============================================================================================================
        # Plot Results
        # ==============================================================================================================
        fig, ax = plt.subplots()
        x_name = 'num_training'
        y_name = 'success_rate'
        ax.plot(df[x_name], df[y_name])
        title = f'Approximate Q Agent - eps={epsilon}, alpha={alpha}, discount={discount}'
        ax.set(xlabel=x_name.replace('_', ' ').upper(), ylabel=y_name.replace('_', ' ').upper(), title=title)
        ax.grid()

        fig.savefig(f"{file_name}.png")

        plt.show()
