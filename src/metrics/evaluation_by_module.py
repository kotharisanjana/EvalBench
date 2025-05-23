import os
from error_handling.custom_error import MetricError, MetricErrorMessages
from utils.registry import metric_registry
from metrics.custom.custom_metrics import load_custom_metrics

def evaluate_module(modules, custom_metric_path = None, **kwargs):
    if not modules:
        raise MetricError(MetricErrorMessages.MISSING_REQUIRED_PARAM, param='module')

    if custom_metric_path:
        if os.path.exists(custom_metric_path):
            load_custom_metrics(custom_metric_path)
        else:
            raise FileNotFoundError(f"Custom metric file not found: {custom_metric_path}")

    results = []
    for name, metric in metric_registry.items():
        if metric.get('module') in modules:
            required_args = metric['required_args']
            try:
                args = {arg: kwargs[arg] for arg in required_args}
                result = metric['func'](**args)
                results.append({'metric': name, 'result': result})
            except Exception as e:
                results.append({'metric': name, 'error': str(e)})

    return results