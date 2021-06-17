# This module defines model-based optimizers used in this project
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from hamiltonian import Hamiltonian
from helpers import w

# Optimizer_LSTM is taken from https://github.com/chenwydj/learning-to-learn-by-gradient-descent-by-gradient-descent
# Optimizer_HNN is adapted based on this code
class Optimizer_LSTM(nn.Module):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0):
        super().__init__()
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs = nn.LSTMCell(2, hidden_sz)
        else:
            self.recurs = nn.LSTMCell(1, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)
        
    def forward(self, inp, hidden, cell):
        if self.preproc:
            # Different input coordinates can have very different magnitudes, 
            # especially with neural networks. To handle it, we have to preprocess
            # inputs as in Appendix A: https://arxiv.org/pdf/1606.04474.pdf  
            inp = inp.data
            inp2 = w(torch.zeros(inp.size()[0], 2))
            keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads]).squeeze()
            
            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = w(Variable(inp2))
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)

class Optimizer_HNN(nn.Module):
    def __init__(self, preproc=False, preproc_factor=10.0):
        super().__init__()
        if preproc:
            gdefunc = Hamiltonian(2)
            self.output = nn.Linear(4, 1, bias=False)
            self.output1 = nn.Linear(4, 1)
        else:
            gdefunc = Hamiltonian(1)
            self.output = nn.Linear(2, 1, bias=False)
            self.output1 = nn.Linear(2, 1)

        self.gde = gdefunc
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)
        
    def forward(self, inp):
        if self.preproc:
            # Different input coordinates can have very different magnitudes, 
            # especially with neural networks. To handle it, we have to preprocess
            # inputs as in Appendix A: https://arxiv.org/pdf/1606.04474.pdf            
            inp_, dev_inp = torch.chunk(inp, 2, dim=-1)
            def preprocess(inp):
                inp = inp.data
                inp2 = w(torch.zeros(inp.size()[0], 2))
                keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
                inp2[:, 0][keep_grads] = (torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
                inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads]).squeeze()
                
                inp2[:, 0][~keep_grads] = -1
                inp2[:, 1][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
                inp = w(Variable(inp2))
                return inp
            # Generalized coordinates and velocities   
            inp_, dev_inp = preprocess(inp_), preprocess(dev_inp)
            inp = torch.cat((inp_, dev_inp), dim=-1)
        # Apply Hamiltonian Neural Network
        h = self.gde(inp)
        deriv = self.output1(h)
        index = int(h.shape[-1]/2)
        return self.output(torch.cat((h[:, :index], inp[:, :index]), dim=-1)), deriv