import os
import psutil
import logging

import openfl.callbacks

logger = logging.getLogger(__name__)

def log_memory_usage(round_num: int, logs=None) -> dict:
    """Logs memory usage details of the current process."""
    del logs  # unused.
    process = psutil.Process(os.getpid())
    virtual_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    memory_usage = {
        "round_num": round_num,
        "process_memory": round(process.memory_info().rss / (1024**2), 2),
        "virtual_memory": {
            "total": round(virtual_memory.total / (1024**2), 2),
            "available": round(virtual_memory.available / (1024**2), 2),
            "percent": virtual_memory.percent,
            "used": round(virtual_memory.used / (1024**2), 2),
            "free": round(virtual_memory.free / (1024**2), 2),
            "active": round(virtual_memory.active / (1024**2), 2),
            "inactive": round(virtual_memory.inactive / (1024**2), 2),
            "buffers": round(virtual_memory.buffers / (1024**2), 2),
            "cached": round(virtual_memory.cached / (1024**2), 2),
            "shared": round(virtual_memory.shared / (1024**2), 2),
        },
        "swap_memory": {
            "total": round(swap_memory.total / (1024**2), 2),
            "used": round(swap_memory.used / (1024**2), 2),
            "free": round(swap_memory.free / (1024**2), 2),
            "percent": swap_memory.percent,
        },
    }
    logger.info(str(memory_usage))

def get_aggregator_callbacks():
    callbacks = []
    callbacks += [openfl.callbacks.TensorBoard(log_dir="./logs/")]
    # callbacks += [openfl.callbacks.LambdaCallback(on_round_end=log_memory_usage)]
    # ckpt_callback = openfl.callbacks.ModelCheckpoint(ckpt_dir="./save/", monitor="val_loss", mode="min", max_to_keep=3)
    return callbacks

def get_collaborator_callbacks():
    callbacks = []
    callbacks += [openfl.callbacks.TensorBoard(log_dir="./logs/")]
    callbacks += [openfl.callbacks.LambdaCallback(on_round_end=log_memory_usage)]
    # ckpt_callback = openfl.callbacks.ModelCheckpoint(ckpt_dir="./save/", monitor="val_loss", mode="min", max_to_keep=3)
    return callbacks