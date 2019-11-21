from .convolution_config import PadConfig


def find_first_before(filter_fn, arr, idx):
    for item in arr[:idx][::-1]:
        if filter_fn(item):
            return item


def parallel_shifted_map(fn, start):
    is_not_pad = lambda node: not isinstance(node, PadConfig)

    def __run__(items):
        return [fn(item, find_first_before(is_not_pad, items, i) if i else start) for i, item in enumerate(items)]
    return __run__
