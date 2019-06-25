import numpy as np
import math

class Node:

    def __init__(self, boardposition, reward, visits):
        self.boardposition = boardposition
        self.reward = reward
        self.visits = visits
        self.children = []

    def add_child(self, boardposition):
        self.children.append(Node(boardposition, 0, 0))

    def isExpanded(self):
        return self.children and all(child.visits > 0 for child in self.children)

    def getPossiblechildren(self, game_state):
        flatgame = np.array(game_state).flatten()
        for position, value in enumerate(flatgame):
            if value == 0:
                self.add_child(position)
        #print(len(self.children))

    def traverse(self, root):
        choosen_child = max(self.children, key= lambda x: x.UCT(root))
        choosen_index = self.children.index(choosen_child)
        return choosen_child, choosen_index
    

    def UCT(self, root):
        '''calculate UCT value for given (leaf) node'''
        if self.visits == 0: return 0
        return self.reward/self.visits + math.sqrt(math.log(root.visits)/self.visits)
        

    def print(self):
        print("Position ", self.boardposition, ", Reward ", self.reward, ", Visits ", self.visits, ", Childcount ", len(self.children))
        print(" ")