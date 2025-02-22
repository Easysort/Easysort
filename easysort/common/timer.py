import time, os

def timeit(name):
    def decorator(f):
        def wrapper(*args, **kwargs):
            if int(os.getenv('DEBUG') or '0') > 0:
                start = time.time()
                result = f(*args, **kwargs)
                print(f"{name} took {time.time() - start:.4f} seconds.")
                return result
            return f(*args, **kwargs)
        return wrapper
    return decorator