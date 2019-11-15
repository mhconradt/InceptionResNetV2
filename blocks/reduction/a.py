"""
Responsible for implementing reduction block A
"""
from torch import nn
from torchinceptionresnetv2.utils import ConvolutionConfig as Convolution
from torchinceptionresnetv2.hparams import K, L, M, N
from torchinceptionresnetv2.blocks.reduction.reduction_block import ReductionBlock
from torchinceptionresnetv2.utils.branch import Branch

INPUT_SIZE = 384


class ReductionA(ReductionBlock):
    def __init__(self):
        left = nn.MaxPool2d(3, padding=0, stride=2)
        center = Branch(INPUT_SIZE, Convolution(N, 3, 2, 'same'))
        right = Branch(INPUT_SIZE, Convolution(K, 1, 1, 'same'), Convolution(L, 3), Convolution(M, 3, 2, 'valid'))
        super().__init__(left, center, right)


if __name__ == '__main__':
    reduction = ReductionA()
    print(reduction)
