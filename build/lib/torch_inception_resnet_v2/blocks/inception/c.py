"""
Responsible for implementing inception block C
"""

from torch import nn
from torch_inception_resnet_v2.blocks.inception.inception_resnet import InceptionResNetBlock
from torch_inception_resnet_v2.utils.branch import Branch
from torch_inception_resnet_v2.utils import ConvolutionConfig as Convolution, PadConfig as Pad


IN_CHANNELS = 1888
SCALE = 0.17


class InceptionC(InceptionResNetBlock):
    def __init__(self):
        left = Branch(IN_CHANNELS, Convolution(192, 1, 1, 0))
        right = Branch(IN_CHANNELS,
                       Convolution(192, 1, 1, 0),
                       Convolution(224, (1, 3), 1, (0, 1)),
                       Convolution(256, (3, 1), 1, (1, 0)))
        join = nn.Conv2d(448, IN_CHANNELS, 1, bias=True)
        super().__init__(1, join, left, right)


if __name__ == '__main__':
    block = InceptionC()
    print(block)
