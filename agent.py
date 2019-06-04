#!/usr/bin/env python3


import torch
import tictactoe as ttt
import random
import matplotlib
import matplotlib.pyplot as plt


board_size = 3
n_spaces = board_size**2
dimensions = [10,9]
learning_rate = 0.05

def activation(z):
	return torch.tanh(z)


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
def get_weights():
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

def get_coords(n,size):

	x = n//size
	y = n%size
	return x,y


def play_episode(Game,agent_parameters,noise = 0.3):

	states_batch = [[],[]]
	actions_batch = [[],[]]
	rewards_batch = [[],[]]
	game_len = 0
	player = False
	over = False
	broke_the_rules = False

	while game_len <= Game.size**2 and not(over):

		if random.random() > noise:
			action = torch.argmax(policy_head(Game.board,agent_parameters[int(player)]))
		else:
			action = torch.tensor(random.randint(0,Game.size**2-1))

		x,y = get_coords(action.item(),Game.size)
		allowed = Game.setField(x,y)

		states_batch[int(player)].append(Game.board)
		actions_batch[int(player)].append(action.item())

		# checks outcome
		res =  Game.checkboard()
		if res:

			rewards_batch[int(player)] = [1.] * len(states_batch[int(player)])
			rewards_batch[int(not(player))] = [-1.] * len(states_batch[int(not(player))])

			over = True

		if not(allowed):

			rewards_batch[int(player)] = [-1.]*len(states_batch[int(player)])
			rewards_batch[int(not(player))] = [0.]*len(states_batch[int(not(player))])

			over = True
			broke_the_rules = True

		# catches draws
		if game_len == Game.size**2:

			rewards_batch[0] = [0.] * len(states_batch[0])
			rewards_batch[1] = [0.] * len(states_batch[1])

			over = True

		game_len += 1
		player = not(player)

	Game.reset()


	return states_batch,actions_batch,rewards_batch,broke_the_rules


def get_training_batch(Game,agent_parameters,batch_size):

	collected = 0

	states = [[],[]]
	actions = [[],[]]
	rewards = [[],[]]

	btr_count= 0

	while collected <= (batch_size//2+Game.size**2):
		states_batch,actions_batch,rewards_batch,btr = play_episode(Game,agent_parameters)

		states[0] += states_batch[0]
		states[1] += states_batch[1]

		actions[0] += actions_batch[0]
		actions[1] += actions_batch[1]

		rewards[0] += rewards_batch[0]
		rewards[1] += rewards_batch[1]

		collected += len(rewards_batch)//2

		btr_count += int(btr)

	complete_states = torch.tensor(states[0][:batch_size//2]+states[1][:batch_size//2])
	complete_actions = torch.tensor(actions[0][:batch_size//2]+actions[1][:batch_size//2])
	complete_rewards = torch.tensor(rewards[0][:batch_size//2]+rewards[1][:batch_size//2])

	return complete_states,complete_actions,complete_rewards,float(btr_count)/(batch_size/2.)


def loss(states,actions,rewards,parameters):

	log_loss = torch.empty(states.size()[0])

	for i in range(states.size()[0]):
		log_loss[i] = rewards[i]*torch.log(policy_head(states[i],parameters)[actions[i]])

	return torch.mean(log_loss)


if __name__ == '__main__':

	Game = ttt.Tictactoe(3,3)
	batch_size = 2000
	no_epochs = 500

	#weight initialization
	parameters = get_weights()
	parameter_pair = [parameters,parameters]

	reward_history = torch.empty(no_epochs)

	for epoch in range(no_epochs):
		states,actions,rewards,btr = get_training_batch(Game,parameter_pair,batch_size)
		l = loss(states,actions,rewards,parameters)
		reward_history[epoch] = torch.mean(rewards)
		print('epoch: '+str(epoch)+
		      '\tloss: '+str('{:5f}'.format(l.item()))+
		      '\treward'+str('{:5f}'.format(torch.mean(rewards).item()))+
		      '\tbroke the rules:'+'{:5f}'.format(btr))
		l.backward()
		with torch.no_grad():
			for layer in range(len(dimensions)):
				parameters[layer][0] -= learning_rate*parameters[layer][0].grad
				parameters[layer][1] -= learning_rate*parameters[layer][1].grad


	plt.plot(reward_history.numpy())
	plt.show()
