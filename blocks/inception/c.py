"""
Responsible for implementing inception block C
"""

from torch import nn
from torchinceptionresnetv2.blocks.inception.inception_resnet import InceptionResNetBlock
from torchinceptionresnetv2.utils.branch import Branch
from torchinceptionresnetv2.utils import ConvolutionConfig as Convolution


IN_CHANNELS = 1888
SCALE = 0.17


class InceptionC(InceptionResNetBlock):
    def __init__(self):
        left = Branch(IN_CHANNELS, Convolution(192, 1))
        right = Branch(IN_CHANNELS, Convolution(192, 1), Convolution(224, (1, 3)), Convolution(256, (3, 1)))
        join = nn.Conv2d(448, 2018, 1, bias=True)
        super().__init__(1, join, left, right)


if __name__ == '__main__':
    block = InceptionC()
    print(block)
