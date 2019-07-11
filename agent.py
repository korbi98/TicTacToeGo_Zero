#!/usr/bin/env python3

import torch
import numpy as np
import tictactoe as ttt
import parameter_managment
import os
import monitoring
import argparse


def activation(z):
	return torch.tanh(z)


class Agent():

	def __init__(self,name):
		self.name = name
		self.parameters,self.metainfo = parameter_managment.load_model(name)


	def infere(self,state):

		# converts to flatt one hot encoding
		in_state = torch.tensor(state)
		flatt = in_state.flatten()

		z = torch.zeros(3,self.metainfo['n_spaces'])
		z[flatt,torch.arange(self.metainfo['n_spaces'])] = 1.
		z = z.flatten()


		# runs through layers
		for layer_index in range(len(self.metainfo['dimensions'])):
			z = activation(self.parameters[layer_index][0]+
				       torch.matmul(self.parameters[layer_index][1],z))

		return z

	# softmax layer for the policy network
	def policy_head(self,state):

		out = self.infere(state)
		exp_out = torch.exp(out)

		return exp_out/torch.sum(exp_out)



	# picks the highest legal option
	# out of a random or a calculated distribution
	def highest_legal(self,state,noise=0.):

		if np.random.random_sample() > noise:
			distribution = self.policy_head(state).numpy()
		else:
			distribution = np.random.random(size=self.metainfo['n_spaces'])

		legal = (np.array(state).flatten()==0).astype(np.int32)
		return np.argmax(legal*distribution)

	# picks the highets option
	# out of a random or caluclated distribution
	# might try to break the rules
	def highest(self,state,noise=0.):
		if np.random.random_sample() > noise:
			distribution = self.policy_head(state).numpy()
		else:
			distribution = np.random.random(size=self.metainfo['n_spaces'])

		return np.argmax(distribution)

	def loss(self,states,actions,rewards):

		batch_size = rewards.size
		log_loss = torch.empty(batch_size)

		for i in range(batch_size):
			log_loss[i] = torch.log(self.policy_head(states[i])[actions[i]])
			log_loss[i] = log_loss[i]*rewards[i]


		return torch.mean(log_loss)

	def optimize(self,lr=None):

		if lr==None:
			lr = self.metainfo['learning_rate']

		with torch.no_grad():
			for layer in range(len(self.metainfo['dimensions'])):
				self.parameters[layer][0] += lr*self.parameters[layer][0].grad
				self.parameters[layer][1] += lr*self.parameters[layer][1].grad
				self.parameters[layer][0].grad.data.zero_()
				self.parameters[layer][1].grad.data.zero_()



def play_episode_(Game,agent_parameters,noise):

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

			rewards_batch[int(player)] = [metainfo['win_reward']] * len(states_batch[int(player)])
			rewards_batch[int(not(player))] = [metainfo['loss_reward']] * len(states_batch[int(not(player))])

			over = 'Win'

		# checks for rule violations
		if not(allowed):

			rewards_batch[int(player)] = [metainfo['cheat_reward']]*len(states_batch[int(player)])
			rewards_batch[int(not(player))] = [0.]*len(states_batch[int(not(player))])

			over = 'Rule Violation'

		# checks for draws
		if game_len == Game.size**2:

			rewards_batch[0] = [metainfo['draw_reward']] * len(states_batch[0])
			rewards_batch[1] = [metainfo['draw_reward']] * len(states_batch[1])

			over = 'Draw'

		game_len += 1
		player = not(player)


	Game.reset()

	return states_batch,actions_batch,rewards_batch,over

def play_episode(Game,agents,metainfo):
	""" 
	Plays on episode of tictactoe gioven an instance of a game
	and a tuple of instaces of agents. Returns a batch of states actions and rewards
	taht the agents recieved as well as the terminating condition.
	"""

	states_batch = [[],[]]
	actions_batch = [[],[]]
	rewards_batch = [[],[]]
	game_len = 0
	player = 0
	over = False

	while game_len <= Game.size**2 and not(over):

		states_batch[player].append(np.array(Game.board))

		action = agents[player].highest_legal(Game.board,noise=metainfo['noise'])
		x,y = Game.get_coords(action.item())
		allowed = Game.setField(x,y)

		actions_batch[player].append(action.item())

		#checks for Win
		res = Game.checkboard()
		if res:
			rewards_batch[player] = [metainfo['win_reward']]*len(states_batch[player])
			rewards_batch[int(player==0)] = [metainfo['loss_reward']]*len(states_batch[int(player==0)])
			over = 'Win'

		#checks for rule violation
		if not(allowed):
			rewards_batch[player] = [metainfo['cheat_reward']]*len(states_batch[player])
			rewards_batch[int(player==0)] = [0.]*len(states_batch[int(player==0)])
			over = 'Rule Violation'

		# checks for draws
		if game_len == Game.size**2:
			rewards_batch[player] = [metainfo['draw_reward']]*len(states_batch[player])
			rewards_batch[int(player==0)] = [metainfo['draw_reward']]*len(states_batch[int(player==0)])
			over = 'Draw'

		player = int(player==0)
		game_len += 1
	Game.reset()
	return states_batch,actions_batch,rewards_batch,over

def get_training_batch(Game,agents,metainfo,batch_size):

	collected = 0

	states = []
	actions = []
	rewards = []
	outcomes = [0,0,0]

	while collected <= batch_size:
		states_batch,actions_batch,rewards_batch,over = play_episode(Game,agents,metainfo)
		if over == 'Rule Violation':
			outcomes[0] += 1.
		elif over == 'Win':
			outcomes[1] += 1.
		elif over == 'Draw':
			outcomes[2] += 1.

		states += states_batch[0]
		actions += actions_batch[0]
		rewards += rewards_batch[0]

		states += states_batch[1]
		actions += actions_batch[1]
		rewards += rewards_batch[1]

		collected += len(rewards_batch[0])+len(rewards_batch[1])

	states = states[:batch_size]
	actions = actions[:batch_size]
	rewards = rewards[:batch_size]
	statistic = np.array(outcomes)
	statistic /= np.linalg.norm(statistic)


	return np.array(states),np.array(actions),np.array(rewards),statistic


def loss_(states,actions,rewards,agent,batch_size):

	log_loss = torch.empty(batch_size)

	for i in range(batch_size):
		log_loss[i] = torch.log(agent.policy_head(states[i])[actions[i]])
		log_loss[i] = log_loss[i]*rewards[i]


	return torch.mean(log_loss)


def train(name):
	subject = Agent(name)
	metainfo = subject.metainfo
	agent_pair = [subject,subject]

	Game = ttt.Tictactoe(metainfo['board_size'], metainfo['win_condition'])

	extra_history = []
	loss_history = []

	Training_Monitor = monitoring.Monitor(metainfo['epoch'])


	while metainfo['epoch'] < metainfo['no_epochs']:


		with torch.no_grad():
			states,actions,rewards,statistic = get_training_batch(Game,agent_pair,metainfo,metainfo['batch_size'])
			extra_history.append(statistic[1])

		l = subject.loss(states,actions,rewards)
		l.backward()
		subject.optimize()

		parameter_managment.save_model(subject.parameters,subject.metainfo,name)

		loss_history.append(l.item())
		print('epoch: '+str(metainfo['epoch'])+
			      '\tloss: '+str('{:5f}'.format(l.item()))+
			      '\textra info: '+str('{:5f}'.format(extra_history[-1])))

		Training_Monitor.refresh(loss_history,extra_history)

		metainfo['epoch'] += 1

def train_pool(name):

	no_members = 20
	if not(os.path.isdir(name)):
		os.mkdir(name)

	member_names = [os.path.join(name,str(i)) for i in range(no_members)]
	pool = [Agent(member_name) for member_name in member_names]
	metainfo = pool[0].metainfo

	Game = ttt.Tictactoe(metainfo['board_size'],metainfo['win_condition'])

	batch_size = metainfo['batch_size']
	no_mini_batches = 20
	mini_batch_size = batch_size//no_mini_batches

	while metainfo['epoch'] < metainfo['no_epochs']:

		p1 = np.random.randint(no_members,size=no_mini_batches)
		p2 = np.random.randint(no_members,size=no_mini_batches)

		with torch.no_grad():

			states = np.empty((batch_size,metainfo['board_size'],metainfo['board_size']),dtype = np.int64)
			actions = np.empty((batch_size),dtype = np.int64)
			rewards = np.empty((batch_size))

			for batch_index in range(0,batch_size,mini_batch_size):
				batchn = batch_index//mini_batch_size
				matchup = (pool[p1[batchn]],pool[p2[batchn]])
				mini_batch = get_training_batch(Game,matchup,metainfo,mini_batch_size)

				states[batch_index:batch_index+mini_batch_size] = mini_batch[0]
				actions[batch_index:batch_index+mini_batch_size] = mini_batch[1]
				rewards[batch_index:batch_index+mini_batch_size] = mini_batch[2]



		result_str = 'epoch: '+str(metainfo['epoch'])

		for member_index in range(no_members):
			l = pool[member_index].loss(states,actions,rewards)
			l.backward()
			pool[member_index].optimize()

			result_str += ' {}: {:3f}'.format(member_index,l.item())
			parameter_managment.save_model(pool[member_index].parameters,
						       metainfo,
						       member_names[member_index])

		print(result_str)


		metainfo['epoch'] += 1



def get_name(dimensions):
	name = 'model'
	for dim in dimensions:
		name += '-'+str(dim)
	return name

def get_args():

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('-model',help='name of the model to be trained')
	parser.add_argument('-train_pool',action='store_true')

	return parser.parse_args()


if __name__ == '__main__':

	args = get_args()
	if args.model != None:
		name = args.model
	else:
		count = 0
		while os.path.isdir('model'+str(count)):
			count += 1
		name = 'model'+str(count)


	if args.train_pool:
		train_pool(name)
	else:
		train(name)


