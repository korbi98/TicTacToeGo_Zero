# Monte Carlo tree search for TicTacToe

import numpy as np
from tictactoe import Tictactoe
import copy
from random import choice
from tree import Node

class MCTS:
    '''
    Class defining a simple monte carlo tree search algorithm.

    Attributes:
        - game_state: the current state of the game (tictactoe board)
        - win_condition: same as win_condition for Tictactoe
        - number_of_rollouts: number of simulations for generating one move
        - tree: list containing all possible and impossible (taken) leaf nodes
        - N: total number of simulations performed since initialisation
    '''
    def __init__(self, game_state, current_player, wincondition, number_of_rollouts):
        self.game_state = game_state
        self.current_player = current_player
        self.tree = Node(-1, 0, 0)
        self.wincondition = wincondition
        
        self.number_of_rollouts = number_of_rollouts
        print(self.game_state)


    def perform_search(self):
        '''Perfoming the mcts by performing the specified number of 
            simulations and updating the corresponding leaf node.
            leaf node is choosen by get_leaf_node function'''
        for i in range(self.number_of_rollouts):
            simulated_game = Tictactoe(len(self.game_state), self.wincondition)
            simulated_game.board = copy.deepcopy(self.game_state)
            simulated_game.move_number = (simulated_game.getFlatgame() != 0).sum()

            leaf, visited_indicies = self.traverse_tree(simulated_game)
            result = self.rollout(simulated_game)
            self.update_tree(result, visited_indicies)
            
        self.tree.print()

        for child in self.tree.children:
            child.print()

        result = [0 for i in range(len(self.game_state)**2)]
        for child in self.tree.children:
            result[child.boardposition] = child.visits
        return result

        
    def update_tree(self, result, visited_indicies):
        '''update leaf in tree'''
        self.tree.visits += 1
        current_node = self.tree
        print(result, visited_indicies)
        for index in visited_indicies:
            current_node = current_node.children[index]
            current_node.visits += 1
            current_node.reward += result
            result = result * -1

            
    def rollout(self, simulated_game):
        '''perform random play for choosen leaf node till terminal
            state is reached'''

        # play random game from leafnode
        while (not simulated_game.checkboard()) and simulated_game.move_number < 9:
            simulated_game.perform_random_move()
    
        simulated_game.printBoard()
        res = simulated_game.checkboard()
        #print(res, self.current_player)
        reward = 0
        if res:
            if res == self.current_player:
                reward = 1
            else: reward = -1
        
        return reward
        

    def traverse_tree(self, simulated_game):
        '''Choose next leaf for performing rollout.
            unexplored leafs are prefered. If parent node is fully
            expanded, leaf with highest UCT value is choosen
        '''
        visited_indicies = []
        current_node = self.tree # root
        while current_node.isExpanded():
            newnode, index = current_node.traverse(self.tree)
            current_node = newnode
            visited_indicies.append(index)
            x,y = simulated_game.get_coords(current_node.boardposition, simulated_game.size)
            simulated_game.setField(x,y)
            
        if not current_node.children:
            current_node.getPossiblechildren(simulated_game.board)

        if not simulated_game.move_number < len(self.game_state)**2:
            return current_node, visited_indicies

        unexplored_leafs = list(filter(lambda x: x.visits == 0, current_node.children))
        leaf = choice(unexplored_leafs)
        visited_indicies.append(current_node.children.index(leaf))
        x,y = simulated_game.get_coords(current_node.boardposition, simulated_game.size)
        simulated_game.setField(x,y)
        return choice(unexplored_leafs), visited_indicies

        
    
