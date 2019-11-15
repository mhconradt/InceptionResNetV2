from torch import nn


class Repeated(nn.Module):
    def __init__(self, module, n):
        super().__init__()
        self.modules = nn.Sequential(*[module() for _ in range(n)])

    def forward(self, x):
        return self.modules(x)
