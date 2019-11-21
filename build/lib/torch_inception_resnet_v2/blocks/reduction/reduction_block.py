"""
Responsible for defining the reduction block
"""
from torch import nn
from torch_inception_resnet_v2.utils import Concurrent


class ReductionBlock(nn.Module):
    def __init__(self, *branches: nn.Module):
        super().__init__()
        self.branches = Concurrent()
        for i, branch in enumerate(branches):
            self.branches.add_module("branch_{}".format(i), branch)

    def forward(self, x):
        return self.branches(x)
