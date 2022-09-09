from time import time


def timer_it(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        print(f'Time! Function {func.__name__!r} executed in {(time()-t1):.4f}s')
        return result
    return wrap_func
