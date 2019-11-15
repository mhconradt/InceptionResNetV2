"""
Responsible for implementing inception block B
"""

from torch import nn
from torchinceptionresnetv2.blocks.inception.inception_resnet import InceptionResNetBlock
from torchinceptionresnetv2.utils.branch import Branch
from torchinceptionresnetv2.utils import ConvolutionConfig as Convolution

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


if __name__ == '__main__':
    block = InceptionB()
    print(block)
