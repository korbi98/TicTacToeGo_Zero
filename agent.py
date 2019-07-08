#!/usr/bin/env python3


import torch
import numpy as np
import tictactoe as ttt
import random
import matplotlib.pyplot as plt
import os

board_size = 3
n_spaces = board_size**2
dimensions = [10,10,n_spaces]
learning_rate = 0.001

def activation(z):
	return torch.tanh(z)
#	return (z>0).to(z)*z

def infere(state,parameters):

	# converts to flatt one hot encoding
	in_state = torch.tensor(state)
	flatt = in_state.flatten()
	z = torch.zeros(3,n_spaces)
	z[flatt,torch.arange(n_spaces)] = 1.
	z = z.flatten()


	# runs through layers
	for layer_index in range(len(dimensions)):
		z = activation(parameters[layer_index][0]+
			       torch.matmul(parameters[layer_index][1],z))

	return z

# softmax layer for the policy network
def policy_head(state,parameters):

	out = infere(state,parameters)
	exp_out = torch.exp(out)

	return exp_out/torch.sum(exp_out)


#weight initialization
def get_random_weights():
	parameters = []

	for i in range(len(dimensions)):
		size = dimensions[i]
		if i == 0:
			prev_size = n_spaces*3
		else:
			prev_size = dimensions[i-1]
		biases = torch.randn(size,requires_grad=True)
		weights = torch.randn((size,prev_size),requires_grad=True)

		parameters.append([biases,weights])

	return parameters

def get_name(name):
	if name == None:
		name = 'savedmodel'
		for dim in dimensions:
			name += '-'+str(dim)
	return name


def save_weights(parameters,name=None):

	name = get_name(name)

	if not(os.path.isdir(name)):
		os.mkdir(name)

	for dim_index in range(len(dimensions)):
		torch.save(parameters[dim_index][0],os.path.join(name,'b'+str(dim_index))+'.pt')
		torch.save(parameters[dim_index][1],os.path.join(name,'w'+str(dim_index))+'.pt')


def load_model(name=None):

	name = get_name(name)

	if not(os.path.isdir(name)):
		return get_random_weights()

	parameters = []
	for dim_index in range(len(dimensions)):
		biases = torch.load(os.path.join(name,'b'+str(dim_index))+'.pt')
		weights = torch.load(os.path.join(name,'w'+str(dim_index))+'.pt')
		parameters.append([biases,weights])

	return parameters

# picks the hioghest legal option
# out of a random or a calculated distribution
def highest_legal(state,parameters,noise=0.):

	if random.random() > noise:
		distribution = policy_head(state,parameters).numpy()
	else:
		distribution = np.random.random(size=n_spaces)

	legal = (np.array(state).flatten()==0).astype(np.int32)
	return np.argmax(legal*distribution)


def highest(state,parameters,noise=0.):
	if random.random() > noise:
		distribution = policy_head(state,parameters).numpy()
	else:
		distribution = np.random.random(size=n_spaces)

	return np.argmax(distribution)



def play_episode(Game,agent_parameters,noise):

	states_batch = [[],[]]
	actions_batch = [[],[]]
	rewards_batch = [[],[]]
	game_len = 0
	player = False
	over = False


	while game_len <= Game.size**2 and not(over):


		states_batch[int(player)].append(np.array(Game.board))

		action = highest_legal(Game.board,agent_parameters[int(player)],noise=noise)
		x,y = Game.get_coords(action.item())
		allowed = Game.setField(x,y)

		actions_batch[int(player)].append(action.item())


		# checks for win
		res =  Game.checkboard()
		if res:

			rewards_batch[int(player)] = [1.] * len(states_batch[int(player)])
			rewards_batch[int(not(player))] = [-1.] * len(states_batch[int(not(player))])

			over = 'Won'

		# checks for rule violations
		if not(allowed):

			rewards_batch[int(player)] = [-5.]*len(states_batch[int(player)])
			rewards_batch[int(not(player))] = [0.]*len(states_batch[int(not(player))])

			over = 'Rule Violation'

		# checks for draws
		if game_len == Game.size**2:

			rewards_batch[0] = [0.] * len(states_batch[0])
			rewards_batch[1] = [0.] * len(states_batch[1])

			over = 'Draw'

		game_len += 1
		player = not(player)


	Game.reset()


	return states_batch,actions_batch,rewards_batch,over


def get_training_batch(Game,agent_parameters,batch_size,noise = 0.0):

	collected = 0

	states = []
	actions = []
	rewards = []
	outcomes = [0,0,0]

	while collected <= batch_size:
		states_batch,actions_batch,rewards_batch,over = play_episode(Game,agent_parameters,noise)
		if over == 'Rule Violation':
			outcomes[0] += 1.
		elif over == 'Won':
			outcomes[1] += 1.
		elif over == 'Draw':
			outcomes[2] += 1.

		states += states_batch[0]
		actions += actions_batch[0]
		rewards += rewards_batch[0]

		states += states_batch[1]
		actions += actions_batch[1]
		rewards += rewards_batch[1]

		collected += len(rewards_batch[0])#+len(rewards_batch[1])

	states = states[:batch_size]
	actions = actions[:batch_size]
	rewards = rewards[:batch_size]
	statistic = np.array(outcomes)
	statistic /= np.linalg.norm(statistic)


	return np.array(states),np.array(actions),np.array(rewards)+10.,statistic


def loss(states,actions,rewards,parameters,batch_size):

	log_loss = torch.empty(batch_size)

	for i in range(batch_size):
		log_loss[i] = torch.log(policy_head(states[i],parameters)[actions[i]])
		log_loss[i] = log_loss[i]*rewards[i]


	return torch.mean(log_loss)

class Monitor():

		def __init__(self):
				plt.ion()
				self.fig = plt.figure(figsize=(6.4,9.))
				self.ax1 = self.fig.add_subplot(211)
				self.ax2 = self.fig.add_subplot(212)
				self.l1, = self.ax1.plot([],'r.')
				self.l2, = self.ax2.plot([])

				self.ax1.set_title('Quotient of games finished')
				self.ax1.set_xlabel('Batches collected')

				self.ax2.set_title('Loss Function')
				self.ax2.set_xlabel('Training Episodes')


		def refresh(self,loss,reward):
				self.l1.set_data(np.arange(len(reward)),reward)
				self.l2.set_data(np.arange(len(loss)),loss)

				self.ax1.autoscale_view()
				self.ax1.relim()
				self.ax2.autoscale_view()
				self.ax2.relim()
				self.fig.canvas.draw()
				plt.pause(.001)


def train():
	Game = ttt.Tictactoe(board_size,3)
	batch_size = 1000
	no_epochs = 100000
	init_noise = 0.9
	noise_decay = 0.9997

	#weight initialization
	parameters = load_model()
	parameter_pair = [parameters,parameters]


	extra_history = []
	loss_history = []


	Training_Monitor = Monitor()


	for epoch in range(no_epochs):

		init_noise *= noise_decay

		with torch.no_grad():
			states,actions,rewards,statistic = get_training_batch(Game,parameter_pair,batch_size,noise = init_noise)
			extra_history.append(init_noise*1)

		l = loss(states,actions,rewards,parameters,batch_size)
		l.backward()
		with torch.no_grad():
			for layer in range(len(dimensions)):
				parameters[layer][0] += learning_rate*parameters[layer][0].grad
				parameters[layer][1] += learning_rate*parameters[layer][1].grad
				parameters[layer][0].grad.data.zero_()
				parameters[layer][1].grad.data.zero_()
				save_weights(parameters)

		loss_history.append(l.item())
		print('epoch: '+str(epoch)+
			      '\tloss: '+str('{:5f}'.format(l.item()))+
			      '\tnoise: '+str('{:5f}'.format(extra_history[-1])))

		Training_Monitor.refresh(loss_history,extra_history)

def color(s,symbol):

	if symbol == 0:
		return '\u001b[31m'+str(s)+'\u001b[0m'
	elif symbol == 1:
		return '\u001b[34m'+str(s)+'\u001b[0m'


def render():


	Game = ttt.Tictactoe(board_size,3)
	parameters = load_model()
	parameter_pair = [parameters,parameters]


	with torch.no_grad():
		for i in range(3):
			states,actions,rewards,statistics = play_episode(Game,parameter_pair,noise=0.3)
			board = []
			counter = 0
			for i in range(3):
				board.append([' ']*3)
			for i in range(len(actions[0])):
				for p in range(2):
					if len(actions[p]) > i:
						coords = Game.get_coords(actions[p][i])
						board[coords[0]][coords[1]] = color(counter,p)
						counter += 1


			print('+-+-+-+')
			for row in board:
				print('|'+row[0]+'|'+row[1]+'|'+row[2]+'|')
				print('+-+-+-+')
			print('')

if __name__ == '__main__':


	train()

