# HUJI AI Projcet - Pommerman
An AI project where we developed AI agents that play the pommerman game.
## Code Structure
Most of this package's code is from the original [pommerman package](https://github.com/MultiAgentLearning/playground).
Our code can be found mainly in the python packages: game_runners and  
pommerman.agents.our_agents, and in smaller changes to some of the original game's code.
 
## Set up the Virtual Environment
1. Create a new virt    ual environment by running: ```python3 -m venv env```.
2. Activate the environment by running ```source ./env/bin/activate.csh``` (or the script appropriate to your system). 
3. Run ```pip install -r requirements.txt``` to install the project's requirements.

### Running games
We prepared some game-runners that can be used to run different configurations of the game. All of them can be 
found in the game_runners file. Each file's code can be edited to configure the game's parameters (such as agent types to use etc, number of games etc.)
A game will end with a tie if it's not over by step 400. This can be configured by changing ```MAX_STEPS``` in ```constants.py```.  
For all files, change the participating agents by uncommenting\commenting the appropriate lines in agents_list. 
Change the depth used in the AlphaBeta agent by changing the DEPTH parameter.

#### 1v1_runner
Runs a 2-playrs game, including versus the alpha-beta agent.
You can change the strategy the agents uses by commenting\uncommenting the appropriate lines and editing agent_list.

#### p_v_c  _runner
Runs a game of human player vs an AlphaBeta agent (or a different agent of your choice). Experience the agony of losing to it yourself!
You can change the strategy the agents uses by commenting\uncommenting the appropriate lines in the agents_list variable.


#### ffa_3_players_runner
Runs a game of 3 players (default - AlphaBeta, simple, smartRandom agents).  
Default depth is 2, which results in a bit slow game but still runs in a reasonable speed.

#### ffa_4_players_runner
Runs a game of 4 players (default - AlphaBeta, simple, simple, smartRandom agents).  
Default depth is 1, which runs a very fast game. Depth can be increase for MUCH better performances at the cost 
of a very long game time 

#### general_runner
A general code meant to be edited according to the desired game configuration.

#### training a Trained Approximate Q-learning Agent
Run trainer_analysis with the appropriate lines edited to your desired configuration. Note that it takes a while to train.
  