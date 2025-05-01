from enum import Enum

class MetricErrorMessages(Enum):
    INVALID_LIST_TYPE = "{param}' must be a list."
    INVALID_INT_TYPE = "{param} must be a positive integer."
    EMPTY_LIST = "{param}' must be a non-empty list."
    EMPTY_STRING = "{param} must be a non-empty string."

    def format_message(self, **kwargs):
        return self.value.format(**kwargs)
