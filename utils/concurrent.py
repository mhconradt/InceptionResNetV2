from torch import nn
import torch
"""
Process multiple modules concurrently
"""


class Concurrent(nn.Sequential):
    def __init__(self, axis=1, stack=False):
        super(Concurrent, self).__init__()
        self.axis = axis
        self.stack = stack

    def forward(self, x):
        outputs = [module(x) for module in self._modules.values()]
        if self.stack:
            return torch.stack(outputs, dim=self.axis)
        return torch.cat(outputs, dim=self.axis)