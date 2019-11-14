from torch import nn

from utils import ConvolutionConfig, parallel_shifted_map

__all__ = ['Branch']

EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.1


class ConvolutionBranchNode(nn.Module):
    def __init__(self, in_filters, config: ConvolutionConfig):
        super().__init__()
        out_filters, kernel_size, stride, padding = config
        # forward, activation scaling and ReLU
        self.convolution = nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding_mode=padding)
        self.bn = nn.BatchNorm2d(num_features=out_filters, eps=EPSILON, momentum=BATCH_NORM_MOMENTUM)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.convolution(x)
        normalized = self.bn(z)
        # apply normalization before activation because that captures the true distribution of activations
        # the mean and standard deviation would be decreased otherwise
        return self.activation(normalized)


class Branch(nn.Sequential):
    def __init__(self, input_size, *args: ConvolutionConfig):
        super().__init__()
        build_nodes = parallel_shifted_map(Branch.node_from_convolution, tuple(input_size))
        self.branches = nn.Sequential(*build_nodes(args))

    def forward(self, x):
        return self.branch(x)

    @staticmethod
    def node_from_convolution(conv, prev_conv):
        return ConvolutionBranchNode(prev_conv[0], conv)