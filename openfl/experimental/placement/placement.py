import functools
import ray

def make_remote(f):
    f = ray.put(f)
    @functools.wraps(f)
    @ray.remote
    def wrapper(*args,**kwargs):
        #args[0] = ray.get(args[0])
        f = getattr(args[0],args[1])
        print(f'\nRunning {f.__name__} in a new process')
        f()
        return args[0]
    return wrapper

def aggregator(f):
    print(f'Aggregator step "{f.__name__}" registered')
    f.task = True
    f.aggregator_step = True
    f.collaborator_step = False
    @functools.wraps(f)
    def wrapper(*args,**kwargs):
        print(f'\nCalling {f.__name__}')
        f(*args,**kwargs)
    return wrapper

def collaborator(f):
    print(f'Collaborator step "{f.__name__}" registered')
    f.task = True
    f.aggregator_step = False
    f.collaborator_step = True
    @functools.wraps(f)
    def wrapper(*args,**kwargs):
        print(f'\nCalling {f.__name__}')
        return f(*args,**kwargs)
    return wrapper

