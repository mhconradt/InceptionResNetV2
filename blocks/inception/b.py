"""
Responsible for implementing inception block B
"""

from torch import nn
from blocks import ConvolutionBranch, InceptionResNetBlock
from utils import ConvolutionConfig as Convolution

IN_CHANNELS = 896
# todo find hyper-parameter SCALE
SCALE = 0.17


class InceptionB(InceptionResNetBlock):
    def __init__(self):
        left = nn.Conv2d(IN_CHANNELS, 192, 1)
        right = ConvolutionBranch(IN_CHANNELS,
                                  Convolution(128, 1),
                                  Convolution(160, (1, 7)),
                                  Convolution(192, (7, 1))
                                  )
        combine = nn.Conv2d(384, IN_CHANNELS, 1, bias=True)
        super().__init__(SCALE, combine, left, right)
