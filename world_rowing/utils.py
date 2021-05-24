
from collections import UserDict, abc
import logging
from typing import Callable, Dict, TypeVar, Tuple, Any
from contextlib import nullcontext
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor, as_completed
)


K = TypeVar("K")
A = TypeVar('A')
V = TypeVar('V')
def map_concurrent(
        func: Callable[..., V], 
        inputs: Dict[K, Tuple], 
        threaded: bool = True, 
        max_workers: int = 10, 
        show_progress: bool = True,
        raise_on_err: bool = False,
        **kwargs, 
    ) -> Tuple[Dict[K, V], Dict[K, Exception]]:
    """
    This function is equalivant to calling,

    >>> output = {k: func(*args, **kwargs) for k, args in inputs.items()}

    except that the function is called using either `ThreadPoolExecutor` 
    if `threaded=True` or a `ProcessPoolExecutor` otherwise.

    The function returns a tuple of `(output, errors)` where errors returns
    the errors that happened during the calling of any of the functions. So
    the function will run all the other work before 

    The function also generates a status bar indicating the progress of the
    computation.

    Alternatively if `raise_on_err=True` then the function will reraise the
    same error.

    Examples
    --------
    >>> import time
    >>> def do_work(arg):
    ...     time.sleep(0.5)
    ...     return arg
    >>> inputs = {i: (i,) for i in range(20)}
    >>> output, errors = map_concurrent(do_work, inputs)
    100%|███████████████████| 20/20 [00:01<00:00, 19.85it/s, completed=18]
    >>> len(output), len(errors)
    (20, 0)

    >>> def do_work2(arg):
    ...     time.sleep(0.5)
    ...     if arg == 5:
    ...         raise(ValueError('something went wrong'))
    ...     return arg
    >>> output, errors = map_concurrent(do_work2, inputs)
    100%|████████| 20/20 [00:01<00:00, 19.86it/s, completed=18, nerrors=1]
    >>> len(output), len(errors)
    (19, 1)
    >>> errors
{5: ValueError('something went wrong')}

    >>> try:
    ...     output, errors = map_concurrent(
    ...         do_work2, inputs, raise_on_err=True)
    ... except ValueError:
    ...     print("task failed successfully!")
    ...
    45%|█████████▍           | 9/20 [00:00<00:00, 17.71it/s, completed=5]
    task failed!
    """
    output = {}
    errors = {}

    Executor = ThreadPoolExecutor if threaded else ProcessPoolExecutor

    if show_progress:
        from tqdm.auto import tqdm
        pbar = tqdm(total=len(inputs)) 
    else:
        pbar = nullcontext()
    with Executor(max_workers=max_workers) as executor, pbar:
        work = {
            executor.submit(func, *args, **kwargs): k
            for k, args in inputs.items()
        }
        status: Dict[str, Any] = {}
        for future in as_completed(work):
            status['completed'] = key = work[future]
            if show_progress:
                pbar.update(1)
                pbar.set_postfix(**status)
            try:
                output[key] = future.result()
            except Exception as exc:
                if raise_on_err:
                    raise exc 
                else:
                    logging.warning(f"{key} experienced error {exc}")
                    errors[key] = exc 
                    status['nerrors'] = len(errors)
    
    return output, errors


def getnesteditem(container, *items):
    value = container
    for item in items:
        value = value[item]
        
    return value


class DictSubset(UserDict):
    def __init__(self, parent, /, subset=(), **kwargs):
        self.parent = parent
        self.data = {}
        self.update(subset)
        self.update(kwargs)
        
    def __getitem__(self, item):
        return self.data[item]
    
    def __setitem__(self, item, value):
        self.data[item] = self.parent[item] = value
        
    def __delitem__(self, item):
        del self.data[item]
        del self.parent[item]
        
    def update(self, other=(), /, **kwargs):
        self.data.update(other, **kwargs)
        self.parent.update(other, **kwargs)
        
    def __repr__(self):
        clsname = self.__class__.__name__
        parent = self.parent
        data = self.data
        return f"{clsname}({parent!r}, subset={data!r})"
    

class AttrDict(UserDict):
    # Start by filling-out the abstract methods
    def __init__(self, data=None, /, **kwargs):
        self.data = DictSubset(self.__dict__)
        self._dict_keys: Dict[str, str] = {}
            
        if data is not None:
            self.update(data)
        if kwargs:
            self.update(kwargs)            
            
    def _mangle(self, name: str) -> str:
        key = name
        for s in '.- ':
            key = key.replace(s, "_")

        if key[0] in '0123456789':
            key = '_' + key
        while key in self.__dict__:
            key += '_data_'

        return key

    def _get_dict_key(self, item):
        if item in self._dict_keys:
            key = self._dict_keys[item]
        else:
            key = self._mangle(item)
            self._dict_keys[item] = key
        
        return key
    
    def __getitem__(self, item):
        return self.__dict__[self._get_dict_key(item)] 
    
    def __setitem__(self, item, value):
        self.__dict__[self._get_dict_key(item)] = value
        
    def update(self, *args, **kwargs):
        return self.data.update(*args, **kwargs)
        
    def __repr__(self):
        clsname = self.__class__.__name__
        data = self.data.data
        return f"{clsname}({data!r})"
    
    @classmethod
    def fromnested(cls, data):
        if isinstance(data, abc.Mapping):
            return cls({
                key: cls.fromnested(value)
                for key, value in data.items()
            })
        elif isinstance(data, list):
            return [cls.fromnested(value) for value in data]
        else:
            return data
    