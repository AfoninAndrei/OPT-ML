# Here, we define the target tasks for optimizees
# Cde is based on: 
# https://github.com/chenwydj/learning-to-learn-by-gradient-descent-by-gradient-descent
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets
from helpers import w

# Our optimizer is supposed to find a 10-element vector, when multiplied by a 10x10 matrix,
# is as close as possible to a 10-element vector. Both vector and matrix are generated randomly 
# from the normal distibution. The error is simply the squared error.
class QuadraticLoss:
    def __init__(self, **kwargs):
        self.W = w(Variable(torch.randn(10, 10)))
        self.y = w(Variable(torch.randn(10)))
        
    def get_loss(self, theta):
        return torch.sum((self.W.matmul(theta) - self.y)**2)

# Basically, it allows sampling batches for training from MNIST dataset
# The loss function is a negative log-likelihood computed in MNISTNet (optimizee.py) 
class MNISTLoss: 
    def __init__(self, training=True):
        try:
            os.mkdir('data')
        except:
            pass
        dataset = datasets.MNIST(
            'data', train=True, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        indices = list(range(len(dataset)))
        np.random.RandomState(10).shuffle(indices)
        if training:
            indices = indices[:len(indices) // 2]
        else:
            indices = indices[len(indices) // 2:]

        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=128,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

        self.batches = []
        self.cur_batch = 0
        
    def sample(self):
        if self.cur_batch >= len(self.batches):
            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch