""" 
Prepares the loading and saving of models.
Creates a random set of weights and metainfo,
if no model is found. Defaults are set at
the beginning of this file.
"""
import yaml
import os
import torch

dimensions= [100,9]

board_size = 3
win_condition = 3

batch_size = 200
learning_rate = 0.01

noise = 0.9
noise_decay = 0.999

no_epochs = 10000
epoch = 0

win_reward = 1.
loss_reward = 0.
cheat_reward = -10.
draw_reward = 0.5

def get_random_weights(metainfo):
	""" 
 	random weight initialization
	"""
	parameters = []

	for i in range(len(metainfo['dimensions'])):
		size = metainfo['dimensions'][i]
		if i == 0:
			prev_size = metainfo['n_spaces']*3
		else:
			prev_size = metainfo['dimensions'][i-1]
		biases = torch.randn(size,requires_grad=True)
		weights = torch.randn((size,prev_size),requires_grad=True)

		parameters.append([biases,weights])

	return parameters



def load_model(name):
	""" 
	loads the model, returns a new/random model if no model is found
	"""

	if not(os.path.isdir(name)):
		metainfo = {'dimensions':dimensions,
                    'board_size':board_size,
                    'n_spaces':board_size**2,
                    'win_condition':win_condition,
                    'batch_size':batch_size,
		    'learning_rate':learning_rate,
                    'noise':noise,
                    'noise_decay':noise_decay,
		    'no_epochs':no_epochs,
                    'epoch':epoch,
		    'win_reward':win_reward,
		    'loss_reward':loss_reward,
		    'cheat_reward':cheat_reward,
		    'draw_reward':draw_reward,
                   }
		return get_random_weights(metainfo),metainfo

	FILE = open(os.path.join(name,'metainfo.txt'),'r')
	metainfo = yaml.safe_load(FILE)
	FILE.close()

	parameters = []
	for dim_index in range(len(dimensions)):
		biases = torch.load(os.path.join(name,'b'+str(dim_index))+'.pt')
		weights = torch.load(os.path.join(name,'w'+str(dim_index))+'.pt')
		parameters.append([biases,weights])

	return parameters,metainfo


def save_model(parameters,metainfo,name):
	""" 
	stores all the relevant information
	"""
	if not(os.path.isdir(name)):
		os.mkdir(name)

	FILE = open(os.path.join(name,'metainfo.txt'),'w')
	metainfo = yaml.dump(metainfo,FILE)
	FILE.close()

	for dim_index in range(len(dimensions)):
		torch.save(parameters[dim_index][0],os.path.join(name,'b'+str(dim_index))+'.pt')
		torch.save(parameters[dim_index][1],os.path.join(name,'w'+str(dim_index))+'.pt')

