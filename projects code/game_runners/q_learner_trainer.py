import argparse
import os
os.environ['LANG']='en_US'
import sys
sys.path.append('.')
import pickle
from time import sleep
import time
from datetime import datetime


import pommerman
from pommerman.agents import QLearningAgent, SimpleAgent, get_identity_features, ApproximateQAgent, \
    AlphaBetaAgent



def parse_args():
    '''CLI interface to bootstrap taining'''
    parser = argparse.ArgumentParser(description="Playgrou====nd Flags.")
    parser.add_argument("--game", default="pommerman", help="Game to choose.")
    parser.add_argument(
        "--config",
        default="PommeFFACompetition-v0",
        help="Configuration to execute. See env_ids in "
             "configs.py for options.")
    parser.add_argument(
        "--agents",
        default="tensorforce::ppo,test::agents.SimpleAgent,"
                "test::agents.SimpleAgent,test::agents.SimpleAgent",
        help="Comma delineated list of agent types and docker "
             "locations to run the agents.")
    parser.add_argument(
        "--agent_env_vars",
        help="Comma delineated list of agent environment vars "
             "to pass to Docker. This is only for the Docker Agent."
             " An example is '0:foo=bar:baz=lar,3:foo=lam', which "
             "would send two arguments to Docker Agent 0 and one to"
             " Docker Agent 3.",
        default="")
    parser.add_argument(
        "--record_pngs_dir",
        default=None,
        help="Directory to record the PNGs of the game. "
             "Doesn't record if None.")
    parser.add_argument(
        "--record_json_dir",
        default=None,
        help="Directory to record the JSON representations of "
             "the game. Doesn't record if None.")
    parser.add_argument(
        "--render",
        default=False,
        action='store_true',
        help="Whether to render or not. Defaults to False.")
    parser.add_argument(
        "--game_state_file",
        default=None,
        help="File from which to load game state. Defaults to "
             "None.")
    args = parser.parse_args()

    # Make the "1v1" environment using the agent list
    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)
    return args


def save_agent(training_agent, prefix, num_games, dir_name="saved_agents"):
    file_name = "{prefix}_{num_games}_{time}.pkl".format(prefix=prefix,
                                                         num_games=num_games,
                                                         time=str(datetime.now()).replace(' ', '_')
                                                            .replace(":", ""))
    pwd = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.join(pwd, dir_name)
    os.makedirs(dir, exist_ok=True)

    path = os.path.join(dir, file_name)
    agent_data = {
        'weights': training_agent.weights,
        'epsilon': training_agent.epsilon,
        'alpha': training_agent.alpha,
        'discount': training_agent.discount,
    }

    with open(path, "bw+") as f:
        pickle.dump(agent_data, f)
    print(f"agent saved into {path}")


def load_agent(filename):
    # pwd = os.path.dirname(os.path.realpath(__file__))
    # full_path = os.path.join(pwd, "saved_agents", filename)
    with open(filename, "rb") as f:
        agent_data = pickle.load(f)
    agent = ApproximateQAgent(extractor="SimpleExtractor")
    agent.weights = agent_data['weights']
    agent.epsilon = agent_data['epsilon']
    agent.alpha = agent_data['alpha']
    agent.discount = agent_data['discount']
    return agent


def train(env, training_agent, num_games, save=False, checkpoint_step=10, dir_name="saved_agents"):
    info = None
    n_wins = 0
    now = time.time()
    for i_episode in range(num_games):
        state = env.reset()
        done = False
        turn = 0
        reward = []
        while not done:
            turn += 1
            # env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            training_agent.observe(state, reward, done, info)

        if i_episode % checkpoint_step == 0 and save:
            save_agent(training_agent, "stats_agent", i_episode, dir_name=dir_name)

        n_wins += 1 if reward[training_agent.id] != -1 else 0
        print(info)
        print(f'Episode {i_episode} finished in {turn} turns')  # TODO print stats ( maybe save to file too?).
    print(f"number of wins during training = {n_wins}, {100.0 * n_wins / num_games}% won")
    print("training took: {}".format(time.time() - now))


def run_games(env, training_agent, num_games, show_games=True):
    training_agent.epsilon = 0
    n_wins = 0
    info = None
    print("run game with the trained agent")
    for i in range(num_games):
        state = env.reset()
        reward = []
        done = False
        while not done:
            if show_games:
                env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            if show_games:
                sleep(0.1)
        if show_games:
            sleep(1)
        print(f"test game num {i} results: {info}")
        n_wins += 1 if reward[training_agent.id] != -1 else 0
    success_rate = 100.0 * n_wins / num_games
    print(f"number of wins during test = {n_wins}, {success_rate}% won")

    env.close()
    return success_rate





def main():
    args = parse_args()


    # ==============================================================================================================
    # Create Agent
    # ==============================================================================================================
    training_agent = ApproximateQAgent(extractor="SimpleExtractor")
    filename = "/Users/tomer/PycharmProjects/AI_project/pommerman/agents/our_agents/rl_agents/trained_agents/trained_q_learners/stats_agent_7900_2020-08-11_04_00_40.355636.pkl"
    training_agent = load_agent(filename)
    # a_b_agent = AlphaBetaAgent()


    # ==============================================================================================================
    # Init Game Enviroment
    # ==============================================================================================================
    a = AlphaBetaAgent(depth=2)
    agent_list = [
        a,
        training_agent,
    ]
    env = pommerman.make('OneVsOne-v0', agent_list)
    training_agent.initialize(env)
    a.initialize(env)
    # a_b_agent.initialize(env)

    # ==============================================================================================================
    # Train Agent
    # ==============================================================================================================
    num_games = 8000
    # train(env, training_agent, num_games=num_games)

    # ==============================================================================================================
    # Save Agent
    # ==============================================================================================================
    # prefix = "arbel_agent"
    # dirname = "saved_agent"
    # save_agent(training_agent, prefix, num_games=num_games, dir_name="saved_2")

    # ==============================================================================================================
    # Run Test Games
    # ==============================================================================================================
    input("ready?")
    run_games(env, training_agent, num_games=100)


if __name__ == "__main__":
    main()
