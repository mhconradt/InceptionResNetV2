from collections import namedtuple


ConvolutionConfig = namedtuple('ConvolutionConfig', ['n_filters', 'kernel_size', 'stride', 'padding'])

PadConfig = namedtuple('PadConfig', ['padding'])
