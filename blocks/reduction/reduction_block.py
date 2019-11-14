"""
Responsible for defining the reduction block
"""
from torch import nn
from utils import Concurrent


class ReductionBlock(nn.Module):
    def __init__(self, *branches: nn.Module):
        super().__init__()
        self.branches = Concurrent()
        for branch in branches:
            self.branches.append(branch)

    def forward(self, x):
        return self.branches(x)
