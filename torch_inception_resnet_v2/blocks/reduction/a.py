"""
Responsible for implementing reduction block A
"""
from torch import nn
from torch_inception_resnet_v2.utils import ConvolutionConfig as Convolution
from torch_inception_resnet_v2.hparams import K, L, M, N
from torch_inception_resnet_v2.blocks.reduction.reduction_block import ReductionBlock
from torch_inception_resnet_v2.utils.branch import Branch

INPUT_SIZE = 384


class ReductionA(ReductionBlock):
    def __init__(self):
        left = nn.MaxPool2d(3, padding=0, stride=2)
        center = Branch(INPUT_SIZE, Convolution(N, 3, 2))
        right = Branch(INPUT_SIZE, Convolution(K, 1, 1, 0), Convolution(L, 3, 1, 1), Convolution(M, 3, 2, 0))
        super().__init__(left, center, right)


if __name__ == '__main__':
    reduction = ReductionA()
    print(reduction)
