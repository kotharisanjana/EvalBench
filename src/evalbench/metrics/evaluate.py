import evalbench
from evalbench.error_handling.custom_error import Error, ErrorMessages

def evaluate_module(module: str, **kwargs):
    if not module:
        raise Error(ErrorMessages.MISSING_REQUIRED_PARAM, param='module')

    if module not in evalbench.module_metric_map:
        raise Error(ErrorMessages.MODULE_NOT_FOUND, param=module)

    results = []
    for metric in evalbench.module_metric_map[module]:
        required_args = metric['required_args']
        try:
            args = {arg: kwargs[arg] for arg in required_args}
            result = metric['func'](**args)
            results.append({'metric': module, 'result': result})
        except Exception as e:
            results.append({'metric': module, 'error': str(e)})

    return results

def evaluate_metric(metric: str, **kwargs):
    if metric not in evalbench.metric_registry.keys():
        raise Error(ErrorMessages.METRIC_NOT_FOUND, param=metric)

    try:
        result = evalbench.metric_registry[metric]['func'](**kwargs)
        return {'metric': metric, 'result': result}
    except Exception as e:
        return {'metric': metric, 'error': str(e)}
