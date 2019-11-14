from torch import nn
from typing import List
from utils.concurrent import Concurrent
"""
Defines the base of an inception ResNet block.
"""


class InceptionResNetBlock(nn.Module):
    def __init__(self, scale, combination: nn.Module, *branches: nn.Module):
        super().__init__()
        self.scale = scale
        self.combination = combination
        self.branches = Concurrent()
        for branch in branches:
            self.branches.append(branch)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        branch_outputs = self.branches(x)
        output = self.combination(branch_outputs)
        res = self.scale * output + x
        return self.activation(res)

