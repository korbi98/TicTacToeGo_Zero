# Defining board, win conditions and actions on board
import numpy as np

class Tictactoe:
    '''
    Class defining the board, checking the winning condition 
    and performing moves on the board

    Attributes:
        - size: defines the size of the quadratic game Board
        - win_condition: defines how many consecutive pieces
             of one player arerequired to win the game
    '''
        
    def __init__(self, size, win_condition):
        assert size >= win_condition, "Size is smaller than win condition"
        self.move_number = 0
        self.size = size
        self.win_condition = win_condition
        self.board = [[0 for i in range(size)] for i in range(size)]


    def setField(self,x,y):
        '''Perform move at x,y. Current Player is determined by move number.
            Returns False if field is already taken'''
        if self.board[x][y] == 0:
            self.board[x][y] = self.move_number % 2 + 1
            self.move_number += 1
            return True
        else:
            return False


    def printBoard(self):
        '''Print board to terminal'''
        for i in range(len(self.board)):
            print(str(i)+": ", end="")
            for j in range(len(self.board[i])):
                print("x" if self.board[j][i] == 1 else "o" if self.board[j][i] == 2 else " ", end ="")
            print()
        print()


    # Checks all possible rows, columns and diagonals
    # returns 1 if player 1 won, 2 if player 2 won, 0 elseflatgame
    def checkboard(self):
        '''Checks all possible row, column and diagonal combinations
            returns 0 if game has not reached terminal state or draw or 
            player that has won'''
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


    def checkArray(self, array):
        '''Checks whether array with lenght of win_condition
            is filled solely by player 1 or 2 and returns that or 0 if not'''
        is_x = [i==1 for i in array]
        if all(is_x):
                return 1

        is_o = [i==2 for i in array]
        if all(is_o):
                return 2

        return 0


    def getRows(self):
        '''returns list of all possible consecutive arrays 
            with length of win_condition for all rows'''
        row_arrays = []
        for i in range(self.size - self.win_condition + 1):
            arrays = self.board[i : i + self.win_condition]
            for j in range(self.size):
                row_arrays.append([arrays[k][j] for k in range(self.win_condition)])
        return row_arrays

 
    def getColumns(self):
        '''same as getRows for columns'''
        column_arrays = []
        for i in self.board:
            for j in range(self.size - self.win_condition + 1):
                column_arrays.append(i[j : j + self.win_condition])
        return column_arrays


    def getDiagonals(self):
        '''same as getRows for diagonals'''
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


    def reset(self):
        '''reset game to initial state'''
        self.board = [[0 for i in range(self.size)] for i in range(self.size)]
        self.move_number = 0


    def perform_random_move(self):
        '''perform random move for current player determined by move_number'''
        flatgame = self.getFlatgame()
        random_policy = np.random.rand(self.size**2)
        random_policy = random_policy * (flatgame == 0)
        choosen_field = np.unravel_index(np.argmax(random_policy), (self.size, self.size))
        self.setField(choosen_field[0], choosen_field[1])


    def getFlatgame(self):
        '''returns flat numpy array (vector) of game board'''
        return np.array(self.board).flatten()


    def get_coords(self,position,size):
        '''return x,y coordinate for flatgame index'''
        x = position//size
        y = position%size
        return x,y