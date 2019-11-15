from torch import nn
from torchinceptionresnetv2.blocks.inception.inception_resnet import InceptionResNetBlock
from torchinceptionresnetv2.utils.branch import Branch
from torchinceptionresnetv2.utils import ConvolutionConfig as Convolution

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


if __name__ == '__main__':
    block = InceptionA()
    print(block)

