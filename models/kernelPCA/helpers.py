import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np
import argparse
import os
import pickle
from PIL import Image
import pandas as pd
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(15)

train_dataset = dsets.MNIST(root='../../utils/data', train=True, download=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='../../utils/data', train=False, download=True, transform=transforms.ToTensor())
# validationloader = DataLoader(dataset = validation_dataset, batch_size = 5000, shuffle = False)

if not (os.path.exists('model/kernel_pca_polynomial.big')):
	print('Invalid Path')
	os._exit(1)

if not (os.path.isfile('model/kernel_pca_polynomial.big')):
	print('Invalid File Name')
	os._exit(1)

store = {}
with open('model/kernel_pca_polynomial.big', 'rb') as f:
	store = pickle.load(f)
kpca = store['kpca']
validation_data = store['validation']
img_dim = validation_data.shape[1]

# Function to plot channels of Convolution Layer
def plot_channels(W, show = False):
	#number of output channels 
	n_out=W.shape[0]
	#number of input channels 
	n_in=W.shape[1]
	w_min=W.min().item()
	w_max=W.max().item()
	fig, axes = plt.subplots(n_out,n_in)
	fig.subplots_adjust(hspace = 0.1)
	out_index=0
	in_index=0
	#plot outputs as rows inputs as columns 
	for ax in axes.flat:
		if in_index>n_in-1:
			out_index=out_index+1
			in_index=0

		ax.imshow(W[out_index,in_index,:,:], vmin=w_min, vmax=w_max, cmap='seismic')
		ax.set_yticklabels([])
		ax.set_xticklabels([])
		in_index=in_index+1

	if show:
		plt.show()

# Function to show data samples
def show_data(dataset,sample, show = False):
    plt.imshow(dataset.x[sample,0,:,:].numpy(),cmap='gray')
    plt.title('y='+str(dataset.y[sample].item()))
    if show:
    	plt.show()

# Create a Data generator
class Data(Dataset):

	# Constructor
	def __init__(self,N_images=100,offset=0,p=0.9, train=False):
		self.x = store['data']
		self.y = train_dataset.targets
		self.len = len(train_dataset)

	def get_val(self, index):
		return (validation_data[index] ,validation_dataset.targets[index])

	def __getitem__(self,index):      
		return self.x[index], self.y[index]
	def __len__(self):
		return self.len


def plot_activations(A,number_rows= 1,name=""):
	A=A[0,:,:,:].detach().numpy()
	n_activations=A.shape[0]

	print(n_activations)
	A_min=A.min().item()
	A_max=A.max().item()

	if n_activations==1:
		# Plot the image.
		plt.imshow(A[0,:], vmin=A_min, vmax=A_max, cmap='seismic')

	else:
		fig, axes = plt.subplots(number_rows, n_activations//number_rows)
		fig.subplots_adjust(hspace = 0.4)
		for i,ax in enumerate(axes.flat):
			if i< n_activations:
				# Set the label for the sub-plot.
				ax.set_xlabel( "activation:{0}".format(i+1))

				# Plot the image.
				ax.imshow(A[i,:], vmin=A_min, vmax=A_max, cmap='seismic')
				ax.set_xticks([])
				ax.set_yticks([])
	plt.show()

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
	from math import floor
	if type(kernel_size) is not tuple:
		kernel_size = (kernel_size, kernel_size)
	h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
	w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
	return h, w

class Neural_Network(nn.Module):

	# Constructor
	def __init__(self, in_dimensions, Hidden1, Hidden2, out_dimensions):
		super(Neural_Network, self).__init__()
		self.linear1 = nn.Linear(in_dimensions, Hidden1)
		self.linear2 = nn.Linear(Hidden1, Hidden2)
		self.linear3 = nn.Linear(Hidden2, out_dimensions)
		# self.linear4 = nn.Linear(Hidden3, Hidden4)
		# self.linear5 = nn.Linear(Hidden4, out_dimensions)

	# Prediction
	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = torch.relu(self.linear1(x))
		x = torch.relu(self.linear2(x))
		# x = torch.relu(self.linear3(x))
		# x = torch.relu(self.linear4(x))
		x = self.linear3(x)
		return x

