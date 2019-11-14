"""
Responsible for implementing inception block C
"""

from torch import nn
from blocks import ConvolutionBranch, InceptionResNetBlock
from utils import ConvolutionConfig as Convolution


IN_CHANNELS = 1792
SCALE = 0.17


class InceptionC(InceptionResNetBlock):
    def __init__(self):
        one = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        two = ConvolutionBranch(IN_CHANNELS, Convolution(256, 1), Convolution(384, 3, 2, 'valid'))
        super().__init__(1, nn.Identity,
                         nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                         ConvolutionBranch(IN_CHANNELS, Convolution(256, 1), Convolution(384, 3, 2, 'valid')),
                         ConvolutionBranch(IN_CHANNELS, Convolution(256, 1), Convolution(288, 3, 2, 'valid')),
                         ConvolutionBranch(IN_CHANNELS, Convolution(256, 1), Convolution(288, 3), Convolution(320, 3, 2, 'valid'))
                         )
