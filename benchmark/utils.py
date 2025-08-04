import psutil
import torch
import time

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)  # in GB
    return 0

def get_cpu_memory():
    return psutil.virtual_memory().percent

def timing(f):
    def wrap(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        duration = time.time() - start
        return result, duration
    return wrap
