class MetricError(Exception):
    def __init__(self, error_message_enum, **kwargs):
        self.message = error_message_enum.format_message(**kwargs)
        super().__init__(self.message)