from error_handling.custom_error import MetricError, MetricErrorMessages

def validate_type_list_non_empty(*args):
    for param_name, arg in args:
        if not isinstance(arg, list) or not arg:
            raise MetricError(MetricErrorMessages.INVALID_LIST, param=param_name)

def validate_type_int_positive_integer(value: int, param_name: str):
    if not isinstance(value, int) or value <= 0:
        raise MetricError(MetricErrorMessages.INVALID_INT, param=param_name)

def validate_type_string_non_empty(*args):
    for param_name, arg in args:
        if not isinstance(arg, str) or not arg:
            raise MetricError(MetricErrorMessages.INVALID_STRING, param=param_name)

def validate_num_args(*args, length: int):
    if len(args[0]) != length:
        raise MetricError(MetricErrorMessages.MISSING_REQUIRED_PARAM)