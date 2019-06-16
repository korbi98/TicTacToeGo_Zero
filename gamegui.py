from tkinter import *
from tkinter import messagebox
import tictactoe as ttt
import numpy as np

# Simple user interface for Game
class Game:
    
    def __init__(self, size, win_condition, isAIstarting = False):
        self.game = ttt.Tictactoe(size, win_condition)
        self.btnGrid = [[0 for i in range(size)] for i in range(size)]
        self.initGui()
        if isAIstarting:
            self.genmove()

    # create grid with size*size buttons
    # set action for each button to btnClick  
    def initGui(self):
        self.root = Tk()
        frame = Frame(self.root)
        self.root.title("TicTacToe")
        frame.grid(row=0, column=0)
        
        for i in range(self.game.size):
            for j in range(self.game.size):
                self.btnGrid[i][j] = Button(frame, text=" ")
                self.btnGrid[i][j].config(height=1, width=1)
                self.btnGrid[i][j].config(font=("Courier", 48, "bold"))
                self.btnGrid[i][j].config(command= lambda x=i, y=j: self.btnClick(x,y))
                self.btnGrid[i][j].grid(column=i, row=j)

        self.root.mainloop()

    # handles buttonclick at position (x,y)
    def btnClick(self, x, y):
        if self.makeMove(x,y):
            self.genmove()

    def makeMove(self, x, y):

        valid = self.game.setField(x,y)
        if valid:
            self.updateGUI()
            self.game.printBoard()
            winner = self.game.checkboard()
            if winner:
                self.showFinishDialog(winner)
                return False
        return valid

    def updateGUI(self):
        for x in range(self.game.size):
            for y in range(self.game.size):
                value = self.game.board[x][y]
                text = "0" if value == 2 else "X" if value == 1 else " "
                self.btnGrid[x][y].config(text=text)

    def resetGame(self):
        for i in self.btnGrid:
            for j in i:
                j.config(text=" ")

    def showFinishDialog(self, winner):
        title = "Player {0} won the game".format(winner)
            
        result = messagebox.askquestion(title, "Start new game?")
        if result == "yes":
            self.game.reset()
            self.resetGame()
        else:
            self.root.destroy()

    def genmove(self):
        # replace np.random with actual policy from network
        flatgame = np.array(self.game.board).T.flatten()
        policy = np.random.rand(self.game.size**2)
        policy = policy * (flatgame == 0)
        bestmove = np.unravel_index(np.argmax(policy), (self.game.size, self.game.size))
        print(policy)
        print(bestmove)
        self.makeMove(bestmove[1], bestmove[0])
        





        