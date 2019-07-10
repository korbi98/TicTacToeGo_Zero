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

def infere(state,parameters):

	# converts to flatt one hot encoding
	in_state = torch.tensor(state)
	flatt = in_state.flatten()

	z = torch.zeros(3,metainfo['n_spaces'])
	z[flatt,torch.arange(metainfo['n_spaces'])] = 1.
	z = z.flatten()


	# runs through layers
	for layer_index in range(len(metainfo['dimensions'])):
		z = activation(parameters[layer_index][0]+
			       torch.matmul(parameters[layer_index][1],z))


	return z

# softmax layer for the policy network
def policy_head(state,parameters):

	out = infere(state,parameters)
	exp_out = torch.exp(out)

	return exp_out/torch.sum(exp_out)



# picks the highest legal option
# out of a random or a calculated distribution
def highest_legal(state,parameters,noise=0.):

	if np.random.random_sample() > noise:
		distribution = policy_head(state,parameters).numpy()
	else:
		distribution = np.random.random(size=metainfo['n_spaces'])

	legal = (np.array(state).flatten()==0).astype(np.int32)
	return np.argmax(legal*distribution)


def highest(state,parameters,noise=0.):
	if np.random.random_sample() > noise:
		distribution = policy_head(state,parameters).numpy()
	else:
		distribution = np.random.random(size=metainfo['n_spaces'])

	return np.argmax(distribution)


def color(s,symbol):

        if symbol == 0:
                return '\u001b[31m'+str(s)+'\u001b[0m'
        elif symbol == 1:
                return '\u001b[34m'+str(s)+'\u001b[0m'

def render():


        Game = ttt.Tictactoe(metainfo['board_size'],metainfo['win_condition'])
        parameter_pair = [parameters,parameters]


        with torch.no_grad():
                for i in range(4):
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
                                str = ''
                                for j in range(len(row)):
                                        str += '|'+row[j]
                                print(str)
                                print('+-+-+-+')
                        print('')

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


def loss(states,actions,rewards,parameters,batch_size):

	log_loss = torch.empty(batch_size)

	for i in range(batch_size):
		log_loss[i] = torch.log(policy_head(states[i],parameters)[actions[i]])
		log_loss[i] = log_loss[i]*rewards[i]


	return torch.mean(log_loss)


def train():
	Game = ttt.Tictactoe(metainfo['board_size'],metainfo['win_condition'])
	noise = metainfo['noise']

	parameter_pair = [parameters,parameters]

	extra_history = []
	loss_history = []

	Training_Monitor = monitoring.Monitor(metainfo['epoch'])


	while metainfo['epoch'] < metainfo['no_epochs']:

		noise *= metainfo['noise_decay']

		with torch.no_grad():
			states,actions,rewards,statistic = get_training_batch(Game,parameter_pair,metainfo['batch_size'],noise = noise)
			extra_history.append(statistic[1])

		l = loss(states,actions,rewards,parameters,metainfo['batch_size'])
		l.backward()

		with torch.no_grad():
			for layer in range(len(metainfo['dimensions'])):
				parameters[layer][0] += metainfo['learning_rate']*parameters[layer][0].grad
				parameters[layer][1] += metainfo['learning_rate']*parameters[layer][1].grad
				parameters[layer][0].grad.data.zero_()
				parameters[layer][1].grad.data.zero_()
				parameter_managment.save_model(parameters,metainfo,name)

		loss_history.append(l.item())
		print('epoch: '+str(metainfo['epoch'])+
			      '\tloss: '+str('{:5f}'.format(l.item()))+
			      '\tnoise: '+str('{:5f}'.format(extra_history[-1])))

		Training_Monitor.refresh(loss_history,extra_history)

		metainfo['epoch'] += 1

def train_pool():

	Game = ttt.Tictactoe(metainfo['board_size'],metainfo['win_condition'])

	batch_size = metainfo['batch_size']
	no_mini_batches = 20
	mini_batch_size = batch_size//no_mini_batches
	lr = metainfo['learning_rate']

	while metainfo['epoch'] < metainfo['no_epochs']:

		p1 = np.random.randint(no_members,size=no_mini_batches)
		p2 = np.random.randint(no_members,size=no_mini_batches)

		with torch.no_grad():

			states = np.empty((batch_size,metainfo['board_size'],metainfo['board_size']),dtype = np.int64)
			actions = np.empty((batch_size),dtype = np.int64)
			rewards = np.empty((batch_size))

			for batch_index in range(0,batch_size,mini_batch_size):
				batchn = batch_index//mini_batch_size
				mini_batch = get_training_batch(Game,
								[pool[p1[batchn]][0],pool[p2[batchn]][0]],
								mini_batch_size,noise = 0.)
				states[batch_index:batch_index+mini_batch_size] = mini_batch[0]
				actions[batch_index:batch_index+mini_batch_size] = mini_batch[1]
				rewards[batch_index:batch_index+mini_batch_size] = mini_batch[2]



		result_str = 'epoch: '+str(metainfo['epoch'])

		for member_index in range(no_members):
			l = loss(states,actions,rewards,pool[member_index][0],batch_size)
			l.backward()

			with torch.no_grad():
				for layer in range(len(metainfo['dimensions'])):
					pool[member_index][0][layer][0] += lr*pool[member_index][0][layer][0].grad
					pool[member_index][0][layer][1] += lr*pool[member_index][0][layer][1].grad
					pool[member_index][0][layer][0].grad.data.zero_()
					pool[member_index][0][layer][1].grad.data.zero_()

			result_str += ' {}: {:3f}'.format(member_index,l.item())
			parameter_managment.save_model(pool[member_index][0],pool[0][1],'train_pool'+str(member_index))

		print(result_str)


		metainfo['epoch'] += 1



def get_name(dimensions):
	name = 'savedmodel'
	for dim in dimensions:
		name += '-'+str(dim)
	return name

def get_args():

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('-model',help='name of the model to be trained')
	parser.add_argument('-train_pool',action='store_true')
	parser.add_argument('-render',action='store_true')

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


	if args.render:
		parameters,metainfo = parameter_managment.load_model(name)
		render()
	elif args.train_pool:
		no_members = 10
		pool = [parameter_managment.load_model('train_pool'+str(i)) for i in range(no_members)]
		metainfo = pool[0][1]
		metainfo['no_epochs'] = 30
		train_pool()
	else:
		print('Training model'+str(name))
		parameters,metainfo = parameter_managment.load_model(name)
		train()


