import importlib
import inspect
from evalbench.utils.config import EvalConfig
from evalbench.utils.runtime import set_config
from evalbench.metrics.evaluate_module import evaluate_module

# import metric modules and individual metrics
_module_names = {
    "response_quality": "metrics.predefined.response_quality",
    "reference_based": "metrics.predefined.reference_based",
    "contextual_generation": "metrics.predefined.contextual_generation",
    "retrieval": "metrics.predefined.retrieval",
    "query_alignment": "metrics.predefined.query_alignment"
}

__all__ = []

for public_name, mod_path in _module_names.items():
    module = importlib.import_module(f"evalbench.{mod_path}")
    globals()[public_name] = module
    __all__.append(public_name)

    for name, obj in inspect.getmembers(module):
        if callable(obj) and not name.startswith("_"):
            globals()[name] = obj
            __all__.append(name)

# additional imports
__all__.extend(['evaluate_module', 'load_custom_metrics'])

# import configs
__all__.extend(["EvalConfig", "set_config"])
