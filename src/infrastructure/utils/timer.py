import time
from contextlib import contextmanager


@contextmanager
def timer(label: str):
    start = time.time()
    try:
        yield
    finally:
        elapsed = (time.time() - start) * 1000
        print(f"{label}: {elapsed:.2f} ms")
