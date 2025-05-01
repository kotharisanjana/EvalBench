from enum import Enum

class MetricErrorMessages(Enum):
    INVALID_LIST_TYPE = "{param}' must be a list."
    INVALID_INT_TYPE = "{param} must be a positive integer."
    EMPTY_LIST = "{param}' must be a non-empty list."
    EMPTY_STRING = "{param} must be a non-empty string."
    MISSING_REQUIRED_PARAM = "{param}' parameter is missing."

    def format_message(self, **kwargs):
        return self.value.format(**kwargs)

class MetricError(Exception):
    def __init__(self, error_message_enum, **kwargs):
        self.message = error_message_enum.format_message(**kwargs)
        super().__init__(self.message)