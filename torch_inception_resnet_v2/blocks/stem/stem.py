"""
Defines the stem of the network
"""
import torch
from torch import nn
from torch_inception_resnet_v2.utils import Branch, ConvolutionConfig as Convolution

IN_SIZE = 3

# a = (a + 2b) - c / d + 1
# d * a = (a + 2b) - c
# d * a + c  = a + 2b
# d * a + c - a = 2b
# ((d * a + c) - a) / 2


class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = Branch(IN_SIZE,
                            Convolution(32, 3, 2, 0),
                            Convolution(32, 3, 1, 0),
                            Convolution(64, 3, 1, 1)
                            )
        self.first_split_left = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.first_split_right = Branch(64, Convolution(96, 3, 2, 0))
        self.second_split_left = Branch(160, Convolution(64, 1, 1, 0), Convolution(96, 3, 1, 0))
        self.second_right = Branch(160,
                                   Convolution(64, 1, 1, 0),
                                   Convolution(64, (7, 1), 1, (3, 0)),
                                   Convolution(64, (1, 7), 1, (0, 3)),
                                   Convolution(96, 3, 1, 0)
                                   )
        self.third_left = Branch(192, Convolution(192, 3, stride=2, padding=0))
        self.third_right = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.first(x)
        left = self.first_split_left(x)
        right = self.first_split_right(x)
        x = torch.cat([left, right], dim=1)
        left = self.second_split_left(x)
        right = self.second_right(x)
        x = torch.cat([left, right], dim=1)
        left = self.third_left(x)
        right = self.third_right(x)
        return torch.cat([left, right], dim=1)


if __name__ == '__main__':
    stem = Stem()
    print(stem)