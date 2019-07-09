# Simple tree structure

import numpy as np
import math

class Node:
    '''
    Class defining a node for the game tree.
    Nodes store their position on the board,
    their reward, their visits and their children.
    '''
    def __init__(self, parent, boardposition, current_player):
        self.parent = parent
        self.boardposition = boardposition
        self.reward = 0
        self.visits = 0
        self.current_player = current_player
        self.children = []

    def add_child(self, boardposition):
        '''add child with certain position on the board'''
        player = 3 - self.current_player
        self.children.append(Node(self, boardposition, player))

    def isExpanded(self):
        '''Check if node is fully expanded, meaning all childs have been visited'''
        return self.children and all(child.visits > 0 for child in self.children)

    def getPossibleChildren(self, game_state):
        '''Used to add all children to node when visited for the first time'''
        flatgame = np.array(game_state).flatten()
        for position, value in enumerate(flatgame):
            if value == 0:
                self.add_child(position)

    def update(self, result):
        self.visits += 1
        reward = 0 # reward a tie with 3
        if result:
            if self.current_player == result:
                reward = 1
            else:
                reward = -1
        self.reward += reward
            # if self.current_player == result:
            #     self.reward[0] += 1 # wins
            # elif self.current_player == 3 - result:
            #     self.reward[1] += 1 # losses

    def UTC_traverse(self, root):
        '''Choosed child for node via UCT function'''
        choosen_child = max(self.children, key= lambda x: x.UCT(root))
        return choosen_child
    

    def UCT(self, root):
        '''calculate UCT value for given (leaf) node'''
        if self.visits == 0: return 0
        return self.reward/self.visits + 1*math.sqrt(math.log(root.visits)/self.visits)

    def print(self, root):
        print("Position ", self.boardposition, ", Player ", self.current_player,\
             ", Reward ", self.reward, ", Visits ", self.visits,\
             ", UTC ", round(self.UCT(root), 3), ", Childcount ", len(self.children))