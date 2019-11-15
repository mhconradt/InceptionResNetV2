"""
Responsible for implementing inception block B
"""

from torch import nn
from .inception_resnet import InceptionResNetBlock
from utils.branch import Branch
from utils import ConvolutionConfig as Convolution

IN_CHANNELS = 896
# todo find hyper-parameter SCALE
SCALE = 0.17


class InceptionB(InceptionResNetBlock):
    def __init__(self):
        left = Branch(IN_CHANNELS, Convolution(192, 1))
        right = Branch(IN_CHANNELS,
                       Convolution(128, 1),
                       Convolution(160, (1, 7)),
                       Convolution(192, (7, 1))
                       )
        combine = nn.Conv2d(384, IN_CHANNELS, 1, bias=True)
        super().__init__(SCALE, combine, left, right)
