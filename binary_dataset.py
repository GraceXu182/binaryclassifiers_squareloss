#from torchvision.datasets import CIFAR10, CIFAR100

import torchvision.datasets
from torch.utils.data import Subset
import random
import torch
import torchvision.transforms as transforms
transform_train = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def getBinaryDS(trainds,class1,class2):
	'''sample two classes (1 and 2) of subsets from training data'''
	# trainds = CIFAR10('~/.torch/data/', train=True, download=True)
	dog_indices, deer_indices, other_indices = [], [], []
	# dog_idx, deer_idx = trainds.class_to_idx['dog'], trainds.class_to_idx['deer']

	dog_idx = class1
	deer_idx = class2

	for i in range(len(trainds)):
		current_class = trainds[i][1]
		if current_class == dog_idx:
			dog_indices.append(i)
		elif current_class == deer_idx:
			deer_indices.append(i)
		else:
			other_indices.append(i)
	
	# dog_indices = dog_indices[:int(0.6 * len(dog_indices))]
	# deer_indices = deer_indices[:int(0.6 * len(deer_indices))]
	#if len(dog_indices)>1000:
	#	dog_indices_1 = dog_indices[:1000]
	#	deer_indices_1 = deer_indices[:1000]
	#else:
	#	dog_indices_1 = dog_indices[:250]
	#	deer_indices_1 = deer_indices[:250]
	#if len(dog_indices)>2000:
	#	dog_indices_1 = dog_indices[:2000]
	#	deer_indices_1 = deer_indices[:2000]
	#else:
	#	dog_indices_1 = dog_indices[:500]
	#	deer_indices_1 = deer_indices[:500]
	trainds_binary = Subset(trainds, dog_indices + deer_indices)
	return trainds_binary

# trainloader, testloader = getCIFAR('CIFAR10',transform_train,transform_train,64,1,2)  
def getCIFAR(dsname,transform_train,transform_test,batch_size,class1,class2):
	# dsname CIFAR10 or CIFAR100
	funcName = getattr(torchvision.datasets, dsname) 
	trainds = funcName('~/.torch/data/', train=True, download=True, transform=transform_train)
	valds   = funcName('~/.torch/data/', train=False, download=True, transform=transform_test)
	
	trainds_binary = getBinaryDS(trainds,class1,class2) 
	valds_binary = getBinaryDS(valds,class1,class2)   
	
	trainloader = torch.utils.data.DataLoader(trainds_binary, batch_size=batch_size,
											  shuffle=True, num_workers=2)
	
	# testset = torchvision.datasets.CIFAR10(root='~/dataset/brainkit/', train=False,
	#                       download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(valds_binary, batch_size=batch_size,
											shuffle=False, num_workers=2)

	return trainloader,testloader


 

