# Here, we implement optimizees that are used in the tasks
# with quadratic functions and MNIST 
# Code is based on: 
# https://github.com/chenwydj/learning-to-learn-by-gradient-descent-by-gradient-descent
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from meta_module import MetaModule, MetaLinear
from helpers import w, to_var

# The optimizee is used for model-based optimizers
class QuadOptimizee(MetaModule):
    def __init__(self, theta=None):
        super().__init__()
        self.register_buffer('theta', to_var(torch.zeros(10).cuda(), requires_grad=True))
        
    def forward(self, target):
        return target.get_loss(self.theta)
    
    def all_named_parameters(self):
        return [('theta', self.theta)]

# Due to unavoidable difficulties, we have to use 
# the same re-implemented optimizee for normal optimizers
class QuadOptimizeeNormal(nn.Module):
    def __init__(self, theta=None):
        super().__init__()
        if theta is None:
            self.theta = nn.Parameter(torch.zeros(10))
        else:
            self.theta = theta
        
    def forward(self, target):
        return target.get_loss(self.theta)
    
    def all_named_parameters(self):
        return [('theta', self.theta)]

# A simple neural network (optimizee) for MNIST task which
# has a single hidden layer with 20 hidden units network.
# We leave the possibility to change easily the activation
# function to experiment with transferring model-based  
# optimizers between different of them
def create_MNISTNet(activation):
    class MNISTNet(MetaModule):
        def __init__(self, layer_size=20, n_layers=1, **kwargs):
            super().__init__()

            inp_size = 28*28
            self.layers = {}
            for i in range(n_layers):
                self.layers[f'mat_{i}'] = MetaLinear(inp_size, layer_size)
                inp_size = layer_size

            self.layers['final_mat'] = MetaLinear(inp_size, 10)
            self.layers = nn.ModuleDict(self.layers)

            self.activation = activation
            self.loss = nn.NLLLoss()

        def all_named_parameters(self):
            return [(k, v) for k, v in self.named_parameters()]

        def forward(self, loss):
            inp, out = loss.sample()
            inp = w(Variable(inp.view(inp.size()[0], 28*28)))
            out = w(Variable(out))

            cur_layer = 0
            while f'mat_{cur_layer}' in self.layers:
                inp = self.activation(self.layers[f'mat_{cur_layer}'](inp))
                cur_layer += 1

            inp = F.log_softmax(self.layers['final_mat'](inp), dim=1)
            l = self.loss(inp, out)
            return l
    return MNISTNet