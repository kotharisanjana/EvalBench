import nltk
import json
import os
from functools import wraps
from typing import Callable, List
from evalbench.utils.registry import metric_registry
from evalbench.utils.runtime import get_config

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

            if cfg.output_mode == 'print':
                _print_results(func.__name__, result, error_message)
            elif cfg.output_mode == 'save':
                _save_results(func.__name__, result, error_message)

            if error_message:
                return {'error': error_message}
            return result
        return wrapper
    return decorator

def _print_results(name, results, error_message=None):
    print(f"\n{name.upper()}:")
    if error_message:
        print(f"Error: {error_message}")
        return

    if not results:
        print("No results")
        return

    if all(isinstance(r, (float, int)) for r in results):
        for i, score in enumerate(results, 1):
            print(f"Score: {score:.3f}")
    elif all(isinstance(r, dict) for r in results):
        for i, res_dict in enumerate(results, 1):
            print(f"Item {i}:")
            for k, v in res_dict.items():
                print(f"    {k}: {v:.3f}")
    else:
        print("Score:", results)

def _save_results(name, result, error_message):
    cfg = get_config()
    directory = os.path.dirname(cfg.json_filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if os.path.exists(cfg.json_filepath):
        with open(cfg.json_filepath, 'r') as f:
            data = json.load(f)
    else:
        data = []

    entry = {
        'metric': name,
    }
    if error_message:
        entry['error'] = error_message
    else:
        entry['result'] = result

    data.append(entry)

    with open(cfg.json_filepath, 'w') as f:
        json.dump(data, f, indent=4)

# Decorator to register metrics with their required arguments
def register_metric(name: str, required_args: List[str], module: str):
    def decorator(func: Callable):
        metric_registry[name] = {
            'func': func,
            'required_args': required_args,
            'module': module,
        }
        return func
    return decorator

# download NLTK data if not present
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt...")
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet')