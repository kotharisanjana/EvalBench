from enum import Enum

class ErrorMessages(Enum):
    # data type checks and validation errors
    INVALID_INT = '{param} must be a positive integer.'
    INVALID_LIST = '{param} must be a non-empty list.'
    INVALID_STRING = '{param} must be a non-empty string.'
    MISSING_REQUIRED_PARAM = 'One/more required parameters missing.'
    LIST_LENGTH_MISMATCH = 'Inputs must be lists of equal length.'

    # integration errors
    METRIC_NOT_FOUND = 'Metric {metric} not found in EvalBench registry.'
    MODULE_NOT_FOUND = 'Module {module} not found in EvalBench registry.'

    # custom errors
    CUSTOM_ERROR = '{message}'

    def format_message(self, **kwargs):
        return self.value.format(**kwargs)

class Error(Exception):
    def __init__(self, error_message_enum, **kwargs):
        self.message = error_message_enum.format_message(**kwargs)
        super().__init__(self.message)