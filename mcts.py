# Monte Carlo tree search for TicTacToe

import numpy as np
from tictactoe import Tictactoe
import copy
from random import choice
from tree import Node
import time

class MCTS:
    '''
    Class defining a simple monte carlo tree search algorithm.

    Attributes:
        - game: instance of TicTacToe game
        - current_player: player to perform next move
        - number_of_rollouts: number of simulations for generating one move
        - tree: list containing all possible and impossible (taken) leaf nodes
    '''
    def __init__(self, game, number_of_rollouts):
        self.game = game
        self.current_player = game.move_number%2 + 1
        print(self.current_player)
        self.tree = Node(None, -1, 3 - self.current_player) # Root node of tree
        self.number_of_rollouts = number_of_rollouts
        print("Initial game state:\n",self.game.board)


    def perform_search(self):
        '''Perfoming the mcts by performing the specified number of 
            simulations and updating the corresponding leaf node.
            leaf node is choosen by traverse_tree function
        '''
        start_time = time.clock()
        for i in range(self.number_of_rollouts):
            simulated_game = copy.deepcopy(self.game)

            # Traverse to leaf
            leaf = self.traverse_tree(simulated_game)
            # Random simulation for leaf
            result = self.rollout(simulated_game)
            # Update all visited nodes 
            self.update_tree(result, leaf)
            
        end_time = time.clock()    

        print("\nFirst layer:")
        for child in self.tree.children:
            child.print(self.tree)

        second_layer = max(self.tree.children, key= lambda x: x.visits)
        print("\nSecond layer:")
        for child in second_layer.children:
            child.print(self.tree)

        print("\nSearch took:", round(end_time-start_time, 4), "seconds")

        result = [0 for i in range(self.game.size**2)]
        for child in self.tree.children:
            result[child.boardposition] = child.visits
        return result
        

    def traverse_tree(self, simulated_game):
        '''Choose next leaf for performing rollout. When node is fully
            expanded, child with highest UCT is choosen. If not a 
            random unexplored node is choosen.
        '''
        current_node = self.tree #root
        while current_node.isExpanded():
            current_node = current_node.UTC_traverse(self.tree)
            x,y = simulated_game.get_coords(current_node.boardposition)
            simulated_game.setField(x,y)

        # create children if empty    
        if not current_node.children:
            current_node.getPossibleChildren(simulated_game.board)

        # terminate if board is full
        if not simulated_game.move_number < simulated_game.size**2 or simulated_game.checkboard():
            return current_node
        
        x,y = simulated_game.get_coords(current_node.boardposition)
        simulated_game.setField(x,y)
        # Choose random unexplored leaf
        unexplored_leafs = list(filter(lambda x: x.visits == 0, current_node.children))
        return choice(unexplored_leafs)


    def rollout(self, simulated_game):
        '''perform random play for choosen leaf node till terminal
            state is reached'''

        while (not simulated_game.checkboard()) and simulated_game.move_number < simulated_game.size**2:
            simulated_game.perform_random_move()
    
        res = simulated_game.checkboard()

        print("Finished simulation player", res, "won. Terminal state is:")
        simulated_game.printBoard()
        return res
        

    def update_tree(self, result, leaf):
        '''update all visited nodes in tree'''

        self.tree.visits += 1
        current_node = leaf

        while current_node.parent:
            #current_node.print(self.tree)
            current_node.update(result)
            current_node = current_node.parent
