from evalbench.runtime_setup.config import EvalConfig, load_config
from evalbench.metrics.evaluate_module import evaluate_module
from evalbench.utils.metrics_helper import expose_metrics, register_metric, handle_output, show_metrics
from evalbench.metrics.custom.custom_metrics import load_custom_metrics
from evalbench.runtime_setup.runtime import set_config
from evalbench.agents.run_agent import run_agent_pipeline

metric_registry = {}
__all__ = []

MODULES = {
    'response_quality': 'metrics.predefined.response_quality',
    'reference_based': 'metrics.predefined.reference_based',
    'contextual_generation': 'metrics.predefined.contextual_generation',
    'retrieval': 'metrics.predefined.retrieval',
    'query_alignment': 'metrics.predefined.query_alignment',
    'response_alignment': 'metrics.predefined.response_alignment',
}

EXPORTED_SYMBOLS = {
    'configs': ['EvalConfig', 'load_config', 'set_config'],
    'module_evaluation': ['evaluate_module'],
    'custom': ['load_custom_metrics'],
    'decorators': ['register_metric', 'handle_output'],
    'agent': ['run_agent_pipeline'],
    'utils': ['show_metrics'],
}

expose_metrics(MODULES)

for group in EXPORTED_SYMBOLS.values():
    __all__.extend(group)
