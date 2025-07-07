import time


def timeit(func):

    def wrapped(*args, **kwargs):

        _start = time.time()

        result = func(*args, **kwargs)

        print(f"Ending `{func.__name__}`; Took {time.time() - _start} seconds")
        print("-" * 50)
        return result

    return wrapped
