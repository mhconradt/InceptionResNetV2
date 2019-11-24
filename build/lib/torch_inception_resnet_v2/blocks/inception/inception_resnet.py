from torch import nn
from torch_inception_resnet_v2.utils.concurrent import Concurrent
"""
Defines the base of an inception ResNet block.
"""


class InceptionResNetBlock(nn.Module):
    def __init__(self, scale, combination: nn.Module, *branches: nn.Module):
        super().__init__()
        self.scale = scale
        self.combination = combination
        self.branches = Concurrent()
        for i, branch in enumerate(branches):
            self.branches.append("branch_{}".format(i), branch)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.branches(x)
        output = self.combination(output)
        output = self.scale * output + x
        return self.activation(output)

