from importlib.util import spec_from_file_location, module_from_spec
import os
import sys

def load_custom_metrics(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Custom metrics file not found: {file_path}")

    module_name = os.path.splitext(os.path.basename(file_path))[0]

    spec = spec_from_file_location(module_name, file_path)
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
