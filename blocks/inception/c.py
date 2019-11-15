"""
Responsible for implementing inception block C
"""

from torch import nn
from .inception_resnet import InceptionResNetBlock
from utils.branch import Branch
from utils import ConvolutionConfig as Convolution


IN_CHANNELS = 1888
SCALE = 0.17


class InceptionC(InceptionResNetBlock):
    def __init__(self):
        left = Branch(IN_CHANNELS, Convolution(192, 1))
        right = Branch(IN_CHANNELS, Convolution(192, 1), Convolution(224, (1, 3)), Convolution(256, (3, 1)))
        join = nn.Conv2d(448, 2018, 1, bias=True)
        super().__init__(1, join, left, right)
