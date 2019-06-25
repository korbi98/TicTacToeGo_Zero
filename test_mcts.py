#!/usr/bin/env python3
import tictactoe as ttt
import mcts

game = ttt.Tictactoe(3,3)
game.perform_random_move()
game.printBoard()

ai = mcts.MCTS(game.board, 2, game.win_condition, 1000)
print(ai.perform_search())
