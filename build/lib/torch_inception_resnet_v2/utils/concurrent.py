from torch import nn
import torch
"""
Process multiple modules concurrently
"""


class Concurrent(nn.Module):
    def __init__(self, axis=1, stack=False):
        super().__init__()
        self.axis = axis
        self.stack = stack
        self.branches = nn.ModuleList([])

    def append(self, module):
        self.branches.append(module)

    def forward(self, x):
        x = [module(x) for module in self.branches]
        if self.stack:
            return torch.stack(x, dim=self.axis)
        return torch.cat(x, dim=self.axis)
