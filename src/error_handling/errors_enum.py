from enum import Enum

class MetricErrorMessages(Enum):
    INVALID_TYPE = "Both '{param1}' and '{param2}' must be lists."
    INVALID_K_TYPE = "'k' must be a positive integer."
    EMPTY_LIST_1 = "{param}' must be a non-empty list."
    EMPTY_LIST_2 = "Both '{param1}' and '{param2}' must be non-empty lists."
    MISSING_REQUIRED_PARAM = "Missing required parameter: '{param}'"
    EMPTY_STRING = "{param} cannot be empty."

    def format_message(self, **kwargs):
        return self.value.format(**kwargs)
