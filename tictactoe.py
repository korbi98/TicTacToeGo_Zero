# Base class for TicTacToe game, defines the game Board
# and winning conditions

import numpy as np

class Tictactoe:

    # size defines the size of the quadratic game Board
    # win_condition defines how many consecutive pieces of one player are
    # required to win the game
    def __init__(self, size, win_condition):
        assert size >= win_condition, "Size is smaller than win condition"
        self.move_number = 0
        self.size = size
        self.win_condition = win_condition
        self.board = [[0 for i in range(size)] for i in range(size)]

    # Perform move at x,y if field is free for player 1 or 2 depending on move_number
    def setField(self,x,y):
        if self.board[x][y] == 0:
            self.board[x][y] = self.move_number % 2 + 1
            self.move_number += 1
            return True
        else:
            return False

    # Print complete board in terminal
    def printBoard(self):
        for i in range(len(self.board)):
            print(str(i)+": ", end="")
            for j in range(len(self.board[i])):
                print("x" if self.board[j][i] == 1 else "o" if self.board[j][i] == 2 else " ", end ="")
            print()
        print()

    # Checks all possible rows, columns and diagonals
    # returns 1 if player 1 won, 2 if player 2 won, 0 else
    def checkboard(self):
        for i in self.getRows():
            res = self.checkArray(i)
            if res:
                return res

        for i in self.getColumns():
            res = self.checkArray(i)
            if res:
                return res

        for i in self.getDiagonals():
            res = self.checkArray(i)
            if res:
                return res

        return 0

    # checks array of size(win_condition)
    def checkArray(self, array):

        is_x = [i==1 for i in array]
        if all(is_x):
                return 1

        is_o = [i==2 for i in array]
        if all(is_o):
                return 2

        return 0

    # returns list of all possible consecutive 
    # arrays of size(win_condition) in all rows
    def getRows(self):
        row_arrays = []
        for i in range(self.size - self.win_condition + 1):
            arrays = self.board[i : i + self.win_condition]
            for j in range(self.size):
                row_arrays.append([arrays[k][j] for k in range(self.win_condition)])
        return row_arrays

    # same as getRows for columns
    def getColumns(self):
        column_arrays = []
        for i in self.board:
            for j in range(self.size - self.win_condition + 1):
                column_arrays.append(i[j : j + self.win_condition])
        return column_arrays

    # same as getRows for diagonals
    def getDiagonals(self):
        diagonals = []
        for i in range(self.size + 1 - self.win_condition):
            for j in range(self.size + 1 - self.win_condition):
                diagonal = []
                for k in range(self.win_condition):
                    diagonal.append(self.board[i+k][j+k])
                
                diagonals.append(diagonal)
                diagonal = []
                for k in range(self.win_condition):
                    diagonal.append(self.board[i+k][j + self.win_condition-1-k])
                    
                diagonals.append(diagonal)
                
        return diagonals

    # reset game to initial state
    def reset(self):
        self.board = [[0 for i in range(self.size)] for i in range(self.size)]
        self.move_number = 0
