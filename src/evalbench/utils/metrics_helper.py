import nltk
import importlib
import inspect
from functools import wraps
from typing import Callable, List, Any
from evalbench.runtime_setup.runtime import get_config
from evalbench.utils.output_control import is_printing_suppressed, print_results, save_results
import evalbench

def _get_input_data(func, args, kwargs):
    from inspect import signature
    sig = signature(func)
    bound = sig.bind(*args, **kwargs)
    return dict(bound.arguments)

def handle_output():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = None
            error_message = None
            cfg = get_config()

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                error_message = str(e)

            input_data = _get_input_data(func, args, kwargs)
            if cfg.output_mode == 'print' and not is_printing_suppressed():
                print_results(func.__name__, input_data, result, error_message)
            elif cfg.output_mode == 'save':
                save_results(func.__name__, input_data, result, error_message)

            if error_message:
                return {'error': error_message}
            return result
        return wrapper
    return decorator

# Decorator to register metrics with their required arguments
def register_metric(name: str, required_args: List[str], arg_types: List[Any], module: str):
    def decorator(func: Callable):
        evalbench.metric_registry[name+'_score'] = {
            'func': func,
            'required_args': required_args,
            'arg_types': arg_types,
            'module': module,
        }
        return func
    return decorator

# expose predefined metrics for package usage
def expose_metrics(module):
    for public_name, module_path in module.items():
        mod = importlib.import_module(f'evalbench.{module_path}')
        setattr(evalbench, public_name, mod)
        if hasattr(evalbench, '__all__'):
            evalbench.__all__.append(public_name)

        for name, obj in inspect.getmembers(mod):
            if callable(obj) and not name.startswith('_'):
                setattr(evalbench, name, obj)
                if hasattr(evalbench, '__all__'):
                    evalbench.__all__.append(name)

# expose metrics module
def expose_custom_metrics(module):
    name = module.__name__.split('.')[-1]
    setattr(evalbench, name, module)
    if hasattr(evalbench, '__all__'):
        evalbench.__all__.append(name)

    for name, obj in inspect.getmembers(module):
        if callable(obj) and not name.startswith('_'):
            setattr(evalbench, name, obj)
            if hasattr(evalbench, '__all__'):
                evalbench.__all__.append(name)

# download NLTK data if not present
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

def show_metrics():
    print("📋 Available Metrics in EvalBench\n")
    print(f"{'Function':<25} {'Module':<20} {'Required Args':<30} {'Arg Types':<40}")
    print("-" * 120)

    for name, meta in evalbench.metric_registry.items():
        func = meta['func'].__name__
        module = meta['module']
        required_args = ", ".join(meta['required_args'])
        arg_types = ", ".join(t.__name__ if hasattr(t, '__name__') else str(t) for t in meta['arg_types'])

        print(f"{func:<25} {module:<20} {required_args:<30} {arg_types:<40}")
