import nltk
import json
import importlib
import inspect
import os
from functools import wraps
from typing import Callable, List, Any
from evalbench.runtime_setup.runtime import get_config
from evalbench.utils.print_control import is_printing_suppressed
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
                _print_results(func.__name__, input_data, result, error_message)
            elif cfg.output_mode == 'save':
                _save_results(func.__name__, input_data, result, error_message)

            if error_message:
                return {'error': error_message}
            return result
        return wrapper
    return decorator

def _print_results(name, input_data, results, error_message=None):
    print(f'\n{name.upper()}:')
    if error_message:
        print(json.dumps({'error': error_message}))
        return

    if not results:
        print(json.dumps({'warning': 'No results'}))
        return

    # Case: results is a list of floats
    if isinstance(results, list) and all(isinstance(r, (float, int)) for r in results):
        for i, score in enumerate(results):
            # Match inputs to corresponding score if inputs are batched
            record = {
                'input': {
                    key: (value[i] if isinstance(value, list) and len(value) == len(results) else value)
                    for key, value in input_data.items()
                },
                'output': score
            }
            print(json.dumps(record))

    # Case: results is a list of dicts
    elif isinstance(results, list) and all(isinstance(r, dict) for r in results):
        for i, res_dict in enumerate(results):
            record = {
                'input': {
                    key: (value[i] if isinstance(value, list) and len(value) == len(results) else value)
                    for key, value in input_data.items()
                },
                'output': res_dict
            }
            print(json.dumps(record))

    # Case: single score or object
    else:
        print(json.dumps({'input': input_data, 'output': results}))

def _save_results(name, input_data, result, error_message):
    cfg = get_config()
    output_path = cfg.output_filepath or f'./outputs/{name}.jsonl'
    directory = os.path.dirname(output_path)

    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(output_path, 'a') as f:
        if error_message:
            record = {
                'metric': name,
                'error': error_message
            }
            f.write(json.dumps(record) + '\n')

        # Case: batched numeric results
        if isinstance(result, list) and all(isinstance(r, (float, int)) for r in result):
            for i, r in enumerate(result):
                record = {
                    'metric': name,
                    'input': {
                        key: (value[i] if isinstance(value, list) and len(value) == len(result) else value)
                        for key, value in input_data.items()
                    },
                    'output': r
                }
                f.write(json.dumps(record) + '\n')

        # Case: batched dict results
        elif isinstance(result, list) and all(isinstance(r, dict) for r in result):
            for i, r in enumerate(result):
                record = {
                    'metric': name,
                    'input': {
                        key: (value[i] if isinstance(value, list) and len(value) == len(result) else value)
                        for key, value in input_data.items()
                    },
                    'output': r
                }
                f.write(json.dumps(record) + '\n')

        # Case: single result
        else:
            record = {
                'metric': name,
                'input': input_data,
                'output': result
            }
            f.write(json.dumps(record) + '\n')

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

def expose_additional_helpers(helpers):
    evalbench.__all__.extend(helpers)

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

def generate_report(request):
    instruction = request.get('instruction', 'N/A').strip()
    task = request.get('task', 'Unknown').strip()
    data = json.dumps(request.get('data', {}), indent=2) if request.get('data') else 'N/A'
    results = request.get('results', {})
    interpretation = request.get('interpretation', '').strip()
    recommendations = request.get('recommendations', '').strip()

    report = [
        '---',
        'LLM Evaluation Report\n',
        f'Instruction:\n{instruction}\n',
        f'Inferred Task:\n{task}\n',
        f'Input Data:\n{data}\n',
    ]

    if results:
        report.append('Evaluation Results:')
        report.append(json.dumps(results, indent=2))
        report.append('')

    if interpretation:
        report.append('Interpretation:')
        report.append(interpretation)
        report.append('')

    if recommendations:
        report.append('Recommendations:')
        report.append(recommendations)
        report.append('')

    report.append('---')
    report.append('This report was generated by EvalBench.')

    return '\n'.join(report)
