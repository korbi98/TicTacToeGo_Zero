#!/usr/bin/env python3
import tictactoe as ttt
import gamegui
import sys

arguments = sys.argv
assert len(arguments) > 2 and len(arguments) < 6, \
    "please provide game size, winning condition, ai mode (optional), \
        number of rollouts for mcts (optional) as argument"


size = int(arguments[1])
winning_condition = int(arguments[2])

if len(arguments) == 4:
    game = gamegui.Game(size, winning_condition, ai_mode=arguments[3])
elif len(arguments) == 5:
    game = gamegui.Game(size, winning_condition, ai_mode=arguments[3], number_of_rollouts = int(arguments[4]))
else:
    game = gamegui.Game(size, winning_condition)
