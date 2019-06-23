# Monte Carlo tree search for TicTacToe

import numpy as np
import math
import tictactoe as ttt
import copy

class MCTS:

    def __init__(self, game_state, current_player, wincondition, number_of_rollouts):
        self.game_state = game_state
        self.current_player = current_player
        self.tree = self.getPosibleActions()
        self.wincondition = wincondition
        self.N = 1
        self.number_of_rollouts = number_of_rollouts
        print(self.game_state)


    def getPosibleActions(self):
        flatgame = np.array(self.game_state).flatten()
        # list with (index, reward, number of visits) if number of visits -1 node is forbidden, because it is not free
        return [[index,0,0] if value == 0 else [index,0,-1] for index, value in enumerate(flatgame)]


    def perform_search(self):
        for i in range(self.number_of_rollouts):
            self.N += 1
            leaf = self.get_leaf_node()
            #print("Perform rollout for ", leaf)
            result = self.rollout(leaf)
            self.update_tree(leaf, result)
            #print(self.tree)
            #print(list(map(lambda x: self.UCT(x), self.tree)))
            #print(" ")

        print(self.tree)
        print(list(map(lambda x: self.UCT(x), self.tree)))
        return list(map(lambda x: x[2], self.tree))

        
    def update_tree(self, leaf, result):
        self.tree[leaf[0]][1] += result
        self.tree[leaf[0]][2] += 1
            
    def rollout(self, leaf):
        game = ttt.Tictactoe(len(self.game_state), self.wincondition)
        game.board = copy.deepcopy(self.game_state)
        flatgame = game.getFlatgame()
        game.move_number = (game.getFlatgame() != 0).sum()

        # make move to leaf
        game.setField(leaf[0]//game.size, leaf[0]%game.size)

        # play random game from leafnode
        while (not game.checkboard()) and game.move_number < 9:
            game.perform_random_move()
    
        res = game.checkboard()

        reward = 0
        if res:
            if res == self.current_player:
                reward = 1
            else: reward = -1
        
        return reward
        

    def get_leaf_node(self):
        unvisited_nodes = list(filter(lambda x: x[2] == 0, self.tree))
        if unvisited_nodes:
            return unvisited_nodes[0]
        else:
            return max(self.tree, key = self.UCT)
        
        

    def UCT(self, node):
        if node[2] < 1: return 0
        return node[1]/node[2] + 2*math.sqrt(math.log(self.N)/node[2])
