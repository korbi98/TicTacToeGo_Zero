""" 
Monitoring of the training process.
One hell of a mess.
"""

import matplotlib.pyplot as plt
import numpy as np

class Monitor():

                def __init__(self,start_epoch):
                                self.start_epoch = start_epoch

                                plt.ion()
                                self.fig = plt.figure(figsize=(6.4,9.))
                                self.ax1 = self.fig.add_subplot(211)
                                self.ax2 = self.fig.add_subplot(212)
                                self.l1, = self.ax1.plot([],'r.')
                                self.l2, = self.ax2.plot([])

                                self.ax1.set_title('Extra Info')
                                self.ax1.set_xlabel('Batches collected')

                                self.ax2.set_title('Loss Function')
                                self.ax2.set_xlabel('Training Episodes')


                def refresh(self,loss,reward):
                                self.l1.set_data(np.arange(len(reward))+self.start_epoch,reward)
                                self.l2.set_data(np.arange(len(loss))+self.start_epoch,loss)

                                self.ax1.autoscale_view()
                                self.ax1.relim()
                                self.ax2.autoscale_view()
                                self.ax2.relim()
                                self.fig.canvas.draw()
                                plt.pause(.001)



class PoolMonitor():

	def __init__(self,start_epoch,no_members):
		self.start_epoch = start_epoch
		self.no_members = no_members

		plt.ion()
		self.fig = plt.figure(figsize=(9.,9.))
		self.ax1 = self.fig.add_subplot(111)
		self.lines = [self.ax1.plot([])[0] for i in range(self.no_members)]

		self.ax1.set_title('Loss of the population')
		self.ax1.set_xlabel('Training Episodes')

	def refresh(self,losses):
		for member_index in range(self.no_members):
			line = self.lines[member_index]
			line.set_data(np.arange(len(losses[member_index]))+self.start_epoch,losses[member_index])

		self.ax1.autoscale_view()
		self.ax1.relim()
		self.fig.canvas.draw()
		plt.pause(0.0001)
