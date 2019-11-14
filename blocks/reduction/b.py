"""
Responsible for implementing reduction block B
"""
from torch import nn
from utils import ConvolutionConfig as Convolution
from utils.branch import Branch
from blocks.reduction import ReductionBlock

IN_CHANNELS = 896


class ReductionB(ReductionBlock):
    def __init__(self):
        one = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        two = Branch(IN_CHANNELS, Convolution(256, 1), Convolution(384, 3, 2, 'valid'))
        three = Branch(IN_CHANNELS, Convolution(256, 1), Convolution(288, 3, 2, 'valid'))
        four = Branch(IN_CHANNELS, Convolution(256, 1), Convolution(288, 3), Convolution(320, 3, 2, 'valid'))
        super().__init__(one, two, three, four)
