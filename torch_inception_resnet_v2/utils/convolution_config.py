from collections import namedtuple


ConvolutionConfig = namedtuple('ConvolutionConfig', ['n_filters', 'kernel_size', 'stride', 'padding'], defaults=[1, 0])

PadConfig = namedtuple('PadConfig', ['padding'])
