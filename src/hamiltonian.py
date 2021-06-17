# Here, we define the physics that is used for HNN optimizer
import torch
from torch import nn

class Hamiltonian(nn.Module):
    def __init__(self, input_dim:int):
        super().__init__()
        self.input_dim = input_dim
        self.g = nn.Sequential(nn.Linear(int(input_dim), int(input_dim)), nn.ReLU(), nn.Linear(int(input_dim), int(input_dim)))
        self.v = nn.Sequential(nn.Linear(int(input_dim), int(input_dim)), nn.ReLU(), nn.Linear(int(input_dim), int(input_dim)))
        self.D = nn.Linear(2*input_dim, 2*input_dim, bias=False)
        self.L = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # Check the report for details about questions below
        with torch.enable_grad():
            q, dev_q = torch.chunk(x, 2, dim=-1)
            g = self.g(q)
            M = torch.matmul(self.L.weight, self.L.weight.t())
            dH_q = self.v(q)
            D_q, D_p = torch.chunk(self.D(torch.cat((dH_q, dev_q), dim=-1)), 2, dim=-1)
            out = torch.cat((dev_q-D_q, torch.matmul(-dH_q-D_p+g, M)), dim=-1).view_as(x)
        return out