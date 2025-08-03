# Neural modules skeleton for HRM+.
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class HModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, plan_dim):
        # initialise GRU and projection here
        pass

    def forward(self, h_prev: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
        # compute new hidden state and plan
        pass

class LModule(nn.Module):
    def __init__(self, plan_dim, hidden_dim):
        # initialise GRU for low-level
        pass

    def converge(self, l_init: Tensor, plan: Tensor, max_steps=10, tol=1e-3) -> Tensor:
        # iteratively refine l_init until convergence
        pass

class MModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        # initialise GRU and projection for middle-level
        pass
