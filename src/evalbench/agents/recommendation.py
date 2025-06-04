from evalbench.runtime_setup.runtime import get_config

class Recommendation:
    def __init__(self, parsed_request):
        self.config = get_config()
        self.parsed_request = parsed_request

