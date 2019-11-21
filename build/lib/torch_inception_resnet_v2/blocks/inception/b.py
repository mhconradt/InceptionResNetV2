"""
Responsible for implementing inception block B
"""

from torch import nn
from torch_inception_resnet_v2.blocks.inception.inception_resnet import InceptionResNetBlock
from torch_inception_resnet_v2.utils.branch import Branch
from torch_inception_resnet_v2.utils import ConvolutionConfig as Convolution

IN_CHANNELS = 896
# todo find hyper-parameter SCALE
SCALE = 0.17


class InceptionB(InceptionResNetBlock):
    def __init__(self):
        left = Branch(IN_CHANNELS, Convolution(192, 1, 1, 0))
        right = Branch(IN_CHANNELS,
                       Convolution(128, 1, 1, 0),
                       Convolution(160, (1, 7), 1, padding=(0, 3)),
                       Convolution(192, (7, 1), 1, padding=(3, 0))
                       )
        combine = nn.Conv2d(384, IN_CHANNELS, 1, bias=True)
        super().__init__(SCALE, combine, left, right)


if __name__ == '__main__':
    block = InceptionB()
    print(block)
