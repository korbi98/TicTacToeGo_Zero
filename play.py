#!/usr/bin/env python3
# Script for launching the game with different parameters
import tictactoe as ttt
import gamegui
import sys

arguments = sys.argv
assert len(arguments) > 3 and len(arguments) < 7, \
    "please provide game size, winning condition, isAIstarting,\
        ai mode (optional),\n number of rollouts for mcts (optional) as argument"


size = int(arguments[1])
winning_condition = int(arguments[2])
isAIstarting = bool(int(arguments[3]))

if len(arguments) == 5:
    assert arguments[4] in ["network", "tree", "random"], \
        "valid options for ai_mode are random, network or tree"
    game = gamegui.Game(size, winning_condition, isAIstarting = isAIstarting, ai_mode=arguments[4])
elif len(arguments) == 6:
    assert arguments[4] in ["network", "tree", "random"], \
        "valid options for ai_mode are random, network or tree"
    game = gamegui.Game(size, winning_condition, isAIstarting = isAIstarting, ai_mode=arguments[4], number_of_rollouts = int(arguments[5]))
else:
    game = gamegui.Game(size, winning_condition, isAIstarting = isAIstarting)
