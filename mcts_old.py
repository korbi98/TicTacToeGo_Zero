import numpy as numpy
import math
from tictactoe import Tictactoe
import copy


class MCTSold:
    '''
    Class defining a simple monte carlo tree search algorithm.
    Only expands first layer after root. Not really a true MCTS
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
        self.tree = self.getPosibleActions()
        self.wincondition = wincondition
        self.N = 0
        self.number_of_rollouts = number_of_rollouts
        print(self.game_state)


    def getPosibleActions(self):
        '''gets all possible actions for current game_state.
            For each field a list containing index, value and visits is generated.
            if visits is set to -1 the field is taken and poses no valid action'''
        flatgame = np.array(self.game_state).flatten()
        return [[index,0,0] if value == 0 else [index,0,-1] for index, value in enumerate(flatgame)]


    def perform_search(self):
        '''Perfoming the mcts by performing the specified number of 
            simulations and updating the corresponding leaf node.
            leaf node is choosen by get_leaf_node function'''
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
        '''update leaf in tree'''
        self.tree[leaf[0]][1] += result
        self.tree[leaf[0]][2] += 1
            
    def rollout(self, leaf):
        '''perform random play for choosen leaf node till terminal
            state is reached'''
        game = Tictactoe(len(self.game_state), self.wincondition)
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
        '''Choose next leaf for performing rollout.
            unexplored leafs are prefered. If parent node is fully
            expanded, leaf with highest UCT value is choosen'''
        unvisited_nodes = list(filter(lambda x: x[2] == 0, self.tree))
        if unvisited_nodes:
            return unvisited_nodes[0]childvisits = list(map(lambda x: x[2], node[3]))
        else:
            return max(self.tree, key = self.UCT)
        
        
    def UCT(self, node):
        '''calculate UCT value for given (leaf) node'''
        if node[2] < 1: return 0
        return node[1]/node[2] + 5*math.sqrt(math.log(self.N)/node[2])
