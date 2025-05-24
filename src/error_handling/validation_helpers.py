from typing import List, Union
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

def validate_list_length(*args):
    if len(args[0]) != len(args[1]):
        raise MetricError(MetricErrorMessages.LIST_LENGTH_MISMATCH)

def validate_batch_inputs(list_1: Union[List[str], List[List[str]]], list_2: Union[List[str], List[List[str]]]):
    validate_num_args((list_1, list_2), length=2)
    validate_list_length(list_1, list_2)
    # for ref, gen in zip(list_1, list_2):
    #     validate_type_string_non_empty(('reference', ref), ('generated', gen))