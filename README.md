# TicTacToeGo_Zero
Maschine Learning with TicTacToe with policy and MCTS (Work in Progress)

The project contains two seperate AIs for playing TicTacToe:

    1. Monte Carlo tree search
    2. Neural Network trained by reinforcement learning (Work in Progress)

To launch the game, simply start the play.py script with the following parameters:

    1. Size of game (e.g. 3 for standard TicTacToe)
    2. Winning condition (number of consecutive pieces)
    3. Start player (0 for Human, 1 for computer)
    4. AI mode (tree, network(only for 3x3 game) or random)(optional, default=network)
    5. Number of Monte Carlo simulations (optional, default=1000)

#### Examples:
    ./play.py 3 3 0
    ./play.py 5 4 1 tree 3000
    ./play.py 6 4 0 random

#### Dependencies:
    - numpy
    - pytorch for network
