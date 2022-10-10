import logging
import os
import sys
import time
from threading import Thread

logger = logging.getLogger(__name__)

class CustomThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                    args=(), kwargs={}):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._result = None
        self._duration = None

    def run(self):
        if self._target is not None:
            start = time.time()
            self._result = self._target(*self._args, **self._kwargs)
            self._exec_duration = round(time.time() - start, 3)

    def result(self):
        return self._result, self._duration

def parameterize(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer        

@parameterize
def fedtiming(func, timeout):
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info(f"Started: ({func.__name__})")

        task = CustomThread(target=func, name=func.__name__, args=args, kwargs=kwargs)
        task.start()
        task.join(timeout)

        result, duration = task.result()

        if duration is None:
            duration = round(time.time() - start, 3)

        logger.info(f"({task.name}) Execution took : {duration} second(s)")
        # logger.debug(f"({task.name}) Execution took : {duration} second(s)")

        if task.is_alive():
            try:
                logger.info(f"Logger Terminating: {task.name}")
                # logger.debug(f"Logger Terminating: {task.name}")
                raise TimeoutError(f"Exception Timeout Terminating: {task.name}")
            except TimeoutError as error:
                print(error)
                print(f"Timeout Terminating: {task.name}", file=sys.stderr) 
            finally:
                os._exit(status=os.EX_TEMPFAIL)
        return result
    return wrapper