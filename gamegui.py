from tkinter import *
from tkinter import messagebox
import tictactoe as ttt

# Simple user interface for Game
class Game:
    
    def __init__(self, size, win_condition):
        self.game = ttt.Tictactoe(size, win_condition)
        self.btnGrid = [[0 for i in range(size)] for i in range(size)]
        self.initGui()

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
        if self.game.setField(x,y):
            text= "O" if self.game.move_number % 2 == 0 else "X"
            self.btnGrid[x][y].config(text=text)
        self.game.printBoard()
        
        winner = self.game.checkboard()
        if winner:
            self.showFinishDialog(winner)

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

        