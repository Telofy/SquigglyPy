from functools import wraps


def aslist(generator):
    @wraps(generator)
    def wrapper(*args, **kwargs):
        return list(generator(*args, **kwargs))

    return wrapper
