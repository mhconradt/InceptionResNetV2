"""
Responsible for implementing reduction block B
"""
from torch import nn
from torch_inception_resnet_v2.utils import ConvolutionConfig as Convolution
from torch_inception_resnet_v2.blocks.reduction.reduction_block import ReductionBlock
from torch_inception_resnet_v2.utils.branch import Branch

IN_CHANNELS = 896


class ReductionB(ReductionBlock):
    def __init__(self):
        one = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        two = Branch(IN_CHANNELS, Convolution(256, 1, 1, 0), Convolution(384, 3, 2, 0))
        three = Branch(IN_CHANNELS, Convolution(256, 1, 1, 0), Convolution(288, 3, 2, 0))
        four = Branch(IN_CHANNELS, Convolution(256, 1, 1, 0), Convolution(288, 3, 1, 1), Convolution(320, 3, 2, 0))
        super().__init__(one, two, three, four)


if __name__ == '__main__':
    reduction = ReductionB()
    print(reduction)
