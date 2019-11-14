"""
Defines the stem of the network
"""
import torch
from torch import nn
from utils import Branch, ConvolutionConfig as Convolution

IN_SIZE = 3


class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = Branch(IN_SIZE, Convolution(32, 3, 2, 'valid'), Convolution(32, 3, 1, 'valid'), Convolution(64, 3))
        self.first_split_left = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.first_split_right = Branch(64, Convolution(96, 3, 2, 'valid'))
        self.second_split_left = Branch(160, Convolution(64, 1), Convolution(96, 3, 1, 'valid'))
        self.second_right = Branch(160,
                                   Convolution(64, 1),
                                   Convolution(64, (7, 1)),
                                   Convolution(64, (1, 7)),
                                   Convolution(96, 3, 1, 'valid')
                                   )
        self.third_left = Branch(192, Convolution(192, 3, padding='valid'))
        self.third_right = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.first(x)
        left = self.first_split_left(x)
        right = self.first_split_right(x)
        # very concerned
        x = torch.cat([left, right], dim=2)
        left = self.second_split_left(x)
        right = self.second_right(x)
        x = torch.cat([left, right], dim=2)
        left = self.third_left(x)
        right = self.third_right(x)
        return torch.cat([left, right], dim=2)