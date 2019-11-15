
def parallel_shifted_map(fn, start):
    def __run__(items):
        return [fn(item, items[i - 1] if i else start) for i, item in enumerate(items)]
    return __run__
