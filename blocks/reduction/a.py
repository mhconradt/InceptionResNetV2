"""
Responsible for implementing reduction block A
"""
from torch import nn
from utils import ConvolutionConfig as Convolution
from hparams import K, L, M, N
from .reduction_block import ReductionBlock
from utils.branch import Branch

INPUT_SIZE = 384


class ReductionA(ReductionBlock):
    def __init__(self):
        left = nn.MaxPool2d(3, padding=0, stride=2)
        center = Branch(INPUT_SIZE, Convolution(N, 3, 2, 'same'))
        right = Branch(INPUT_SIZE, Convolution(K, 1, 1, 'same'), Convolution(L, 3), Convolution(M, 3, 2, 'valid'))
        super().__init__(left, center, right)
