from tkinter import *
from tkinter import messagebox
import tictactoe as ttt

class Game:
    def __init__(self, size, win_condition):
        self.game = ttt.Tictactoe(size, win_condition)
        self.btnGrid = [[0 for i in range(size)] for i in range(size)]
        self.initGui()
        
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

    def btnClick(self, x, y):
        if self.game.board[x][y] == 0:
            text= "X" if self.game.move_number % 2 == 0 else "O"
            self.btnGrid[x][y].config(text=text)
            self.game.setField(x,y)
        self.game.printField()
        if self.game.checkboard():
            self.showFinishDialog()

    def resetGame(self):
        for i in self.btnGrid:
            for j in i:
                j.config(text=" ")

    def showFinishDialog(self):
        msg = ""
        if self.game.move_number % 2 == 1:
            msg = "Player 1 won the game"
        else:
            msg = "Player 2 won the game"
            
        result = messagebox.askquestion(msg, "Start new game?")
        if result == "yes":
            self.game.reset()
            self.resetGame()
        else:
            self.root.destroy()

        