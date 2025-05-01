from error_handling.custom_error import MetricError
from error_handling.errors_enum import MetricErrorMessages

def validate_list_type_and_non_empty(*args):
    for param_name, arg in args:
        if not isinstance(arg, list):
            raise MetricError(MetricErrorMessages.INVALID_LIST_TYPE, param=param_name)
        if not arg:
            raise MetricError(MetricErrorMessages.EMPTY_LIST, param=param_name)

def validate_positive_integer(value: int, param_name: str):
    if not isinstance(value, int) or value <= 0:
        raise MetricError(MetricErrorMessages.INVALID_INT_TYPE, param=param_name)

def validate_string_non_empty(*args):
    for param_name, arg in args:
        raise MetricError(MetricErrorMessages.EMPTY_STRING, param=param_name)

