from tkinter import *
from tkinter import messagebox
import tictactoe as ttt
import numpy as np
import agent
import mcts

# Simple user interface for Game
class Game:
    
    def __init__(self, size, win_condition, isAIstarting = False, ai_mode = "network", number_of_rollouts = 500):
        self.game = ttt.Tictactoe(size, win_condition)
        self.btnGrid = [[0 for i in range(size)] for i in range(size)]
        self.isAistarting = isAIstarting
        self.ai_mode = ai_mode
        self.number_of_rollouts = number_of_rollouts
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

        if self.isAistarting:
            self.genmove()

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
            if winner or self.game.move_number >= self.game.size**2:
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
        if self.isAistarting:
            self.genmove()

    def showFinishDialog(self, winner):
        title = ("Player " + str(winner) if winner != 0 else "Nobody") + " has won"
            
        result = messagebox.askquestion(title, "Start new game?")
        if result == "yes":
            self.game.reset()
            self.resetGame()
        else:
            self.root.destroy()

    def genmove(self):
        
        flatgame = self.game.getFlatgame()
        policy = np.zeros(self.game.size**2)
        
        if (self.ai_mode == "network"):
            policy = agent.policy_head(self.game.board, agent.get_weights()).detach().numpy()
            policy = policy * (flatgame == 0)
        elif (self.ai_mode == "tree"):
            current_player = self.game.move_number % 2 + 1
            tree_ai = mcts.MCTS(self.game.board, current_player, self.game.win_condition, self.number_of_rollouts)
            policy = tree_ai.perform_search()
        elif (self.ai_mode == "random"):
            policy = np.random.rand(self.game.size**2)
            policy = policy * (flatgame == 0)

        bestmove = np.unravel_index(np.argmax(policy), (self.game.size, self.game.size))
        print(policy)
        print(bestmove)
        self.makeMove(bestmove[0], bestmove[1])
        