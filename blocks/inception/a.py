from torch import nn
from blocks import InceptionResNetBlock
from utils.branch import Branch
from utils import ConvolutionConfig as Convolution

"""
Responsible for implementing inception block A
"""

IN_CHANNELS = 384
SCALE = 0.17


class InceptionA(InceptionResNetBlock):
    def __init__(self):
        right = Branch(IN_CHANNELS,
                       Convolution(32, 1),
                       Convolution(48, 3),
                       Convolution(64, 3)
                       )
        middle = Branch(IN_CHANNELS,
                        Convolution(32, 1),
                        Convolution(32, 3)
                        )
        left = Branch(IN_CHANNELS,
                      Convolution(32, 1)
                      )
        combine = nn.Conv2d(128, out_channels=IN_CHANNELS, kernel_size=1, bias=True)
        super().__init__(SCALE, combine, left, middle, right)
