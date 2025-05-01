import json
import os
from functools import wraps
from evalbench_metrics import config
from typing import Callable, List

def handle_output():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, suppress_output=False, **kwargs):
            result = func(*args, **kwargs)
            if not suppress_output:
                if config.output_mode == 'print':
                    _print_results(func.__name__, result)
                elif config.output_mode == 'save':
                    _save_results(func.__name__, result)
            return result
        return wrapper
    return decorator

def _print_results(name, result):
    print(f"\n{name.upper()} Evaluation Result:")
    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for subk, subv in v.items():
                    print(f"    {subk}: {subv:.3f}")
            else:
                print(f"  {k}: {v:.3f}")
    else:
        print(f"  Score: {result:.3f}")


def _save_results(name, result):
    directory = os.path.dirname(config.json_filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if os.path.exists(config.json_filepath):
        with open(config.json_filepath, 'r') as f:
            data = json.load(f)
    else:
        data = []

    entry = {
        'metric': name,
        'result': result,
    }
    data.append(entry)

    with open(config.json_filepath, 'w') as f:
        json.dump(data, f, indent=4)

# Decorator to register metrics with their required arguments
def register_metric(name: str, required_args: List[str]):
    metric_registry = {}
    def decorator(func: Callable):
        metric_registry[name] = {
            'func': func,
            'required_args': required_args
        }
        return func
    return decorator