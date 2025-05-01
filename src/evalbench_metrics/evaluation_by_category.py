from typing import Any

from error_handling.custom_error import MetricError, MetricErrorMessages
from utils.registry import metric_registry

def evaluate_module(category: str, **kwargs) -> list[Any]:
    if not category:
        raise MetricError(MetricErrorMessages.MISSING_REQUIRED_PARAM, param='category')

    results = []
    for name, metric in metric_registry.items():
        if metric.get('category') == category:
            required_args = metric['required_args']
            try:
                args = {arg: kwargs[arg] for arg in required_args}
                result = metric['func'](**args)
                results.append({'metric': name, 'result': result})
            except KeyError as e:
                results.append({'metric': name, 'error': str(e)})

    return results