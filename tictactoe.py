
class Tictactoe:

    def __init__(self, size, win_condition):
        assert size >= win_condition, "Size is smaller than win condition"
        self.move_number = 0
        self.size = size
        self.win_condition = win_condition
        self.board = [[0 for i in range(size)] for i in range(size)]

    def setField(self,x,y):
        self.board[x][y] = self.move_number % 2 + 1
        self.move_number += 1

    def printField(self):
        for i in range(len(self.board)):
            print(str(i)+": ", end="")
            for j in range(len(self.board[i])):
                print("x" if self.board[j][i] == 1 else "o" if self.board[j][i] == 2 else " ", end ="")
            print()
        print()

    def checkboard(self):
        for i in range(self.size):
            if self.checkArray(self.getRow(i)):
                return True

        for i in range(self.size):
            if self.checkArray(self.getColumn(i)):
                return True
        for i in self.getDiagonals():
            if self.checkArray(i):
                return True

        return False

    def checkArray(self, array):
        #print(array)
        currentNumber = -1
        counter = 1
        enoughInRow = False
        for i in array:
            if i != 0 and i == currentNumber:
                counter += 1
            else:
                counter = 1
            currentNumber = i
            if counter >= self.win_condition:
                return True
        return False

    
    def getColumn(self, index):
        return self.board[index]

    def getRow(self, index):
        column = []
        for i in range(self.size):
            column.append(self.board[i][index])
        return column

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

    def reset(self):
        self.board = [[0 for i in range(self.size)] for i in range(self.size)]
        self.move_number = 0